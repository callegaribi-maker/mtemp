"""
TUG em lote (multiplos pacientes)
==================================

Pagina adicional do Momentum Web, criada para permitir a analise do TUG de
varios pacientes de uma so vez, sem alterar em nada o app original
(`mWEB.py`) nem o modulo de processamento (`processamento/tugProcessing.py`).

Como funciona
-------------
1. Envie, de uma so vez, os arquivos de acelerometro e giroscopio de todos os
   pacientes (o par de cada paciente e identificado automaticamente pelo
   nome do arquivo, no padrao usado pelo app: "<Paciente> tug_acelData_...txt"
   e "<Paciente> tug_gyroData_...txt").
2. Para cada paciente, roda-se exatamente o mesmo processamento usado na
   analise individual (mesmos filtros, mesma logica de inicio/fim do teste).
3. O resultado sai em bloco: uma tabela com uma linha por paciente e as
   mesmas variaveis que o app exporta no modo individual, pronta para
   baixar em CSV/TXT.
4. Para conferir visualmente cada paciente (inicio/fim do teste e os dois
   picos de giro marcados), cada paciente tem um expansor com os graficos
   da aceleracao resultante e da velocidade angular resultante.

Esta pagina aparece automaticamente no menu lateral do Streamlit (pasta
`pages/`) quando o app e implantado a partir deste fork -- o arquivo
`mWEB.py` continua sendo o ponto de entrada e nao precisa de nenhuma
alteracao.
"""

import io
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from processamento.tug_metrics import processar_paciente_tug

st.set_page_config(page_title="TUG em lote - Momentum Web", page_icon="🧍", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🧍 TUG — Análise em lote</h1>",
            unsafe_allow_html=True)
st.caption(
    "Envie os arquivos de acelerômetro e giroscópio de vários pacientes de uma vez. "
    "O pareamento acc/gyro de cada paciente é feito automaticamente pelo nome do arquivo."
)


def carregar_dados_generico(nome, raw_bytes):
    """Mesma logica de leitura usada em mWEB.py (carregar_dados_generico),
    adaptada para operar sobre bytes ja lidos de um UploadedFile."""
    head = raw_bytes[:32]
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        encodings = ["utf-16"]
    else:
        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    df = None
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python", encoding=enc)
            last_err = None
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err

    if df.shape[1] == 5:
        dados = df.iloc[:, 1:5].copy()
    elif df.shape[1] == 4:
        dados = df.iloc[:, 0:4].copy()
    else:
        raise ValueError(f"{nome}: o arquivo deve conter 4 ou 5 colunas com cabeçalhos.")
    dados.columns = ["Tempo", "X", "Y", "Z"]
    return dados


def identificar_paciente(nome_arquivo: str):
    """Extrai (nome_do_paciente, tipo) a partir do padrao de nome de arquivo
    usado pelo Momentum ('<Paciente> tug_acelData_...' / '... tug_gyroData_...').
    Retorna tipo = 'acel', 'gyro' ou None se nao reconhecido."""
    m = re.search(r"(.*?)\s*tug_(acel|gyro)Data", nome_arquivo, re.IGNORECASE)
    if not m:
        return nome_arquivo, None
    paciente = m.group(1).strip()
    tipo = "acel" if m.group(2).lower() == "acel" else "gyro"
    return paciente, tipo


def _y_no_instante(t_array, y_array, t_alvo):
    """Valor de y_array no ponto de t_array mais proximo de t_alvo — usado só
    para posicionar visualmente um marcador de pico sobre a curva resultante
    (o pico em si é sempre detectado pelo tugProcessing original, sem
    alteração; isto é apenas para desenhar o marcador no lugar certo)."""
    idx = int(np.argmin(np.abs(np.asarray(t_array) - t_alvo)))
    return y_array[idx]


def plotar_resultantes(sinais):
    """Dois gráficos de conferência por paciente, usando apenas os sinais
    resultantes (norma): aceleração e velocidade angular, com início/fim do
    teste e os dois picos de giro (3 m e 6 m) marcados."""
    t_acc = sinais["t_novo_acc"]
    norma_acc = sinais["norma_acc_filtrado"]
    t_gyro = sinais["t_novo_gyro"]
    norma_gyro = sinais["norma_gyro_filtrado"]
    start_test = sinais["start_test"]
    stop_test = sinais["stop_test"]

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 3.2))
        ax1.plot(t_acc, norma_acc, linewidth=0.8, color="black")
        ax1.axvline(start_test, color="green", linestyle="--", label="Início", linewidth=0.9)
        ax1.axvline(stop_test, color="red", linestyle="--", label="Final", linewidth=0.9)
        ax1.set_xlim(start_test - 5, stop_test + 5)
        ax1.set_xlabel("Tempo (s)")
        ax1.set_ylabel("Aceleração resultante (m/s²)")
        ax1.legend(fontsize=8)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
        ax2.plot(t_gyro, norma_gyro, linewidth=0.8, color="black")
        ax2.axvline(start_test, color="green", linestyle="--", label="Início", linewidth=0.9)
        ax2.axvline(stop_test, color="red", linestyle="--", label="Final", linewidth=0.9)
        y_g1 = _y_no_instante(t_gyro, norma_gyro, sinais["G1_lat"])
        y_g2 = _y_no_instante(t_gyro, norma_gyro, sinais["G2_lat"])
        ax2.plot(sinais["G1_lat"], y_g1, "ro", label="Giro 3 m")
        ax2.plot(sinais["G2_lat"], y_g2, "ro", label="Giro 6 m")
        ax2.set_xlim(start_test - 5, stop_test + 5)
        ax2.set_xlabel("Tempo (s)")
        ax2.set_ylabel("Vel. angular resultante (rad/s)")
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


st.subheader("1) Importar arquivos")
arquivos = st.file_uploader(
    "Selecione TODOS os arquivos (acelerômetro + giroscópio) de TODOS os pacientes",
    type=["csv", "txt"],
    accept_multiple_files=True,
)

st.subheader("2) Parâmetros do processamento (aplicados a todos os pacientes)")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    baseline_onset_global = st.number_input("Baseline inicial (s)", value=0.0, step=0.1)
with col_b:
    margem_final = st.number_input("Margem antes do fim do registro (s)", value=2.5, step=0.1,
                                    help="Baseline final = duração total do registro − esta margem, "
                                         "igual ao padrão usado na análise individual.")
with col_c:
    filtro_acc = st.number_input("Filtro passa-baixa acelerômetro (Hz)", value=2.0, step=0.1)
with col_d:
    filtro_gyro = st.number_input("Filtro passa-baixa giroscópio (Hz)", value=1.25, step=0.05)

if arquivos:
    pares = {}
    nao_reconhecidos = []
    for arq in arquivos:
        paciente, tipo = identificar_paciente(arq.name)
        if tipo is None:
            nao_reconhecidos.append(arq.name)
            continue
        pares.setdefault(paciente, {})[tipo] = arq

    if nao_reconhecidos:
        st.warning(
            "Os arquivos abaixo não seguem o padrão esperado de nome "
            "('<Paciente> tug_acelData_...' / '<Paciente> tug_gyroData_...') e foram ignorados:\n\n"
            + "\n".join(f"- {n}" for n in nao_reconhecidos)
        )

    incompletos = [p for p, v in pares.items() if "acel" not in v or "gyro" not in v]
    if incompletos:
        st.warning(
            "Os pacientes abaixo têm apenas um dos dois arquivos (acelerômetro OU giroscópio) "
            "e serão ignorados até que o par completo seja enviado:\n\n"
            + "\n".join(f"- {p}" for p in incompletos)
        )

    completos = {p: v for p, v in pares.items() if "acel" in v and "gyro" in v}

    if completos:
        st.success(f"{len(completos)} paciente(s) prontos para processar: "
                   + ", ".join(sorted(completos.keys())))

        if st.button("▶️ Processar todos os pacientes", type="primary"):
            linhas = []
            sinais_por_paciente = {}
            erros = []
            barra = st.progress(0.0)
            for i, (paciente, arqs) in enumerate(sorted(completos.items())):
                try:
                    dados_acc = carregar_dados_generico(arqs["acel"].name, arqs["acel"].getvalue())
                    dados_gyro = carregar_dados_generico(arqs["gyro"].name, arqs["gyro"].getvalue())

                    baseline_offset = (np.max(dados_acc["Tempo"]) / 1000) - margem_final

                    resultado = processar_paciente_tug(
                        dados_acc, dados_gyro,
                        baseline_onset_global, baseline_offset,
                        filtro_acc=filtro_acc, filtro_gyro=filtro_gyro,
                    )
                    metrics = resultado["metrics"]
                    sinais_por_paciente[paciente] = resultado["sinais"]

                    # sinaliza casos que provavelmente precisam de checagem manual
                    # (mesma checagem que um analista faria olhando os gráficos
                    # no modo individual: fases não podem ter duração negativa)
                    fases = [
                        metrics.get("Duração para levantar (s)", metrics.get("Duracao para levantar (s)")),
                        metrics.get("Duração da caminhada de ida (s)", metrics.get("Duracao da caminhada de ida (s)")),
                        metrics.get("Duração da caminhada de volta (s)", metrics.get("Duracao da caminhada de volta (s)")),
                        metrics.get("Duração para sentar (s)", metrics.get("Duracao para sentar (s)")),
                    ]
                    alerta = "Revisar (fase com duração negativa)" if any(f < 0 for f in fases) else ""

                    linhas.append({"Paciente": paciente, **metrics, "Alerta": alerta})
                except Exception as e:
                    erros.append((paciente, str(e)))
                barra.progress((i + 1) / len(completos))

            if erros:
                st.error("Falha ao processar os pacientes abaixo:\n\n"
                          + "\n".join(f"- {p}: {e}" for p, e in erros))

            if linhas:
                resultado_df = pd.DataFrame(linhas)
                st.session_state["tug_lote_resultado"] = resultado_df
                st.session_state["tug_lote_sinais"] = sinais_por_paciente

    # Exibe resultado + gráficos se já processado (mantém após reruns de widgets)
    if "tug_lote_resultado" in st.session_state:
        resultado_df = st.session_state["tug_lote_resultado"]
        sinais_por_paciente = st.session_state["tug_lote_sinais"]

        st.subheader("3) Resultado em bloco")
        if (resultado_df["Alerta"] != "").any():
            st.info(
                "Pacientes marcados com alerta tiveram alguma fase do TUG com duração "
                "negativa — isso indica que os picos identificados provavelmente não "
                "correspondem às transições reais do teste. Confira o gráfico desse "
                "paciente abaixo antes de usar os valores."
            )
        st.dataframe(resultado_df, use_container_width=True)

        csv_bytes = resultado_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 Baixar resultados em lote (.csv)",
            data=csv_bytes,
            file_name="resultados_TUG_lote.csv",
            mime="text/csv",
        )

        txt_bytes = resultado_df.to_csv(index=False, sep="\t").encode("utf-8")
        st.download_button(
            "📄 Baixar resultados em lote (.txt, tab-separado)",
            data=txt_bytes,
            file_name="resultados_TUG_lote.txt",
            mime="text/plain",
        )

        st.subheader("4) Conferir início/fim e picos de cada paciente")
        st.caption(
            "Cada gráfico mostra o sinal resultante (norma) — verde/vermelho marcam "
            "início e fim do teste, os pontos vermelhos marcam os dois picos de giro "
            "(3 m e 6 m). Expanda um paciente para conferir."
        )
        for paciente in sorted(sinais_por_paciente.keys()):
            alerta_paciente = resultado_df.loc[resultado_df["Paciente"] == paciente, "Alerta"].iloc[0]
            titulo = f"👤 {paciente}" + (" ⚠️" if alerta_paciente else "")
            with st.expander(titulo):
                plotar_resultantes(sinais_por_paciente[paciente])
    else:
        st.info("Envie ao menos um par completo (acelerômetro + giroscópio) de um paciente.")
else:
    st.info("Aguardando o envio dos arquivos.")
