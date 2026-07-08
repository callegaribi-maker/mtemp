"""
TUG em lote (multiplos pacientes) — versao standalone
========================================================

Este arquivo roda o mesmo processamento de TUG do mWEB.py (via
`processamento/tugProcessing.py`, sem nenhuma alteracao), mas permite
processar varios pacientes de uma vez e conferir/corrigir cada um.

Por que existe correcao manual por paciente
---------------------------------------------
O algoritmo original identifica o inicio/fim do teste com um limiar fixo
dentro de uma janela de "baseline", e os dois picos de giro (3 m e 6 m)
pegando simplesmente os dois maiores picos da velocidade angular vertical em
todo o registro -- sem nenhuma restricao de que estejam dentro do teste.
Isso significa que, se houver algum movimento maior fora da janela do teste
(ajuste do sensor, etc.), o algoritmo pode marcar o pico errado. No app
individual isso exigia inspecao visual caso a caso; aqui, cada paciente tem
campos para corrigir manualmente o inicio, o fim e os dois picos de giro,
recalculando as metricas derivadas na hora -- sem alterar a deteccao
automatica em si (`tugProcessing.processar_tug` continua intocado).
"""

import io
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from processamento.tug_metrics import processar_paciente_tug, metricas_a_partir_de_sinais

st.set_page_config(page_title="TUG em lote - Momentum Web", page_icon="🧍", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🧍 TUG — Análise em lote</h1>",
            unsafe_allow_html=True)
st.caption(
    "Envie os arquivos de acelerômetro e giroscópio de vários pacientes de uma vez. "
    "O pareamento acc/gyro de cada paciente é feito automaticamente pelo nome do arquivo."
)


def carregar_dados_generico(nome, raw_bytes):
    """Mesma lógica de leitura usada em mWEB.py (carregar_dados_generico),
    adaptada para operar sobre bytes já lidos de um UploadedFile."""
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
    m = re.search(r"(.*?)\s*tug_(acel|gyro)Data", nome_arquivo, re.IGNORECASE)
    if not m:
        return nome_arquivo, None
    paciente = m.group(1).strip()
    tipo = "acel" if m.group(2).lower() == "acel" else "gyro"
    return paciente, tipo


def _y_no_instante(t_array, y_array, t_alvo):
    idx = int(np.argmin(np.abs(np.asarray(t_array) - t_alvo)))
    return y_array[idx]


def _metrics_para_linha(paciente, metrics, corrigido):
    fases = [
        metrics["Duracao para levantar (s)"],
        metrics["Duracao da caminhada de ida (s)"],
        metrics["Duracao da caminhada de volta (s)"],
        metrics["Duracao para sentar (s)"],
    ]
    alerta = "Revisar (fase com duração negativa)" if any(f < 0 for f in fases) else ""
    return {"Paciente": paciente, **metrics, "Corrigido manualmente": "Sim" if corrigido else "",
            "Alerta": alerta}


def plotar_resultantes(sinais):
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

st.subheader("2) Parâmetros iniciais (aplicados a todos os pacientes ao processar)")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    baseline_onset_global = st.number_input("Baseline inicial (s)", value=0.0, step=0.1)
with col_b:
    margem_final = st.number_input("Margem antes do fim do registro (s)", value=2.5, step=0.1,
                                    help="Baseline final = duração total do registro − esta margem, "
                                         "igual ao padrão usado na análise individual.")
with col_c:
    filtro_acc_global = st.number_input("Filtro passa-baixa acelerômetro (Hz)", value=2.0, step=0.1)
with col_d:
    filtro_gyro_global = st.number_input("Filtro passa-baixa giroscópio (Hz)", value=1.25, step=0.05)

if "tug_lote_dados" not in st.session_state:
    st.session_state["tug_lote_dados"] = {}

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
            erros = []
            barra = st.progress(0.0)
            for i, (paciente, arqs) in enumerate(sorted(completos.items())):
                try:
                    dados_acc = carregar_dados_generico(arqs["acel"].name, arqs["acel"].getvalue())
                    dados_gyro = carregar_dados_generico(arqs["gyro"].name, arqs["gyro"].getvalue())
                    baseline_offset = (np.max(dados_acc["Tempo"]) / 1000) - margem_final

                    resultado = processar_paciente_tug(
                        dados_acc, dados_gyro, baseline_onset_global, baseline_offset,
                        filtro_acc=filtro_acc_global, filtro_gyro=filtro_gyro_global,
                    )
                    st.session_state["tug_lote_dados"][paciente] = {
                        "dados_acc": dados_acc,
                        "dados_gyro": dados_gyro,
                        "sinais": resultado["sinais"],
                        "metrics": resultado["metrics"],
                        "corrigido": False,
                    }
                except Exception as e:
                    erros.append((paciente, str(e)))
                barra.progress((i + 1) / len(completos))

            if erros:
                st.error("Falha ao processar os pacientes abaixo:\n\n"
                          + "\n".join(f"- {p}: {e}" for p, e in erros))

    dados_processados = {
        p: v for p, v in st.session_state["tug_lote_dados"].items() if "metrics" in v
    }

    if dados_processados:
        st.subheader("3) Resultado em bloco")
        linhas = [
            _metrics_para_linha(p, v["metrics"], v.get("corrigido", False))
            for p, v in sorted(dados_processados.items())
        ]
        resultado_df = pd.DataFrame(linhas)

        if (resultado_df["Alerta"] != "").any():
            st.info(
                "Pacientes marcados com alerta tiveram alguma fase do TUG com duração "
                "negativa — sinal de que o início/fim ou os picos de giro não foram bem "
                "identificados automaticamente. Abra o paciente na seção 4 para corrigir."
            )
        st.dataframe(resultado_df, use_container_width=True)

        csv_bytes = resultado_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 Baixar resultados em lote (.csv)",
            data=csv_bytes, file_name="resultados_TUG_lote.csv", mime="text/csv",
        )
        txt_bytes = resultado_df.to_csv(index=False, sep="\t").encode("utf-8")
        st.download_button(
            "📄 Baixar resultados em lote (.txt, tab-separado)",
            data=txt_bytes, file_name="resultados_TUG_lote.txt", mime="text/plain",
        )

        st.subheader("4) Conferir e corrigir cada paciente")
        st.caption(
            "Verde/vermelho = início e fim do teste detectados automaticamente. Pontos "
            "vermelhos = os dois picos de giro (3 m e 6 m). Se algum desses 4 instantes "
            "não estiver no lugar certo, digite o valor correto (em segundos, lendo no "
            "próprio gráfico) e clique em aplicar — as métricas do paciente são "
            "recalculadas na hora, sem mexer na detecção automática dos outros pontos."
        )
        for paciente in sorted(dados_processados.keys()):
            info = dados_processados[paciente]
            alerta_paciente = resultado_df.loc[resultado_df["Paciente"] == paciente, "Alerta"].iloc[0]
            titulo = f"👤 {paciente}" + (" ⚠️" if alerta_paciente else "")
            with st.expander(titulo):
                plotar_resultantes(info["sinais"])

                st.markdown("**Corrigir manualmente (opcional)**")
                sinais = info["sinais"]
                cc1, cc2, cc3, cc4 = st.columns(4)
                with cc1:
                    novo_inicio = st.number_input(
                        "Início (s)", value=float(sinais["start_test"]),
                        step=0.05, key=f"start_{paciente}",
                    )
                with cc2:
                    novo_fim = st.number_input(
                        "Fim (s)", value=float(sinais["stop_test"]),
                        step=0.05, key=f"stop_{paciente}",
                    )
                with cc3:
                    novo_g1 = st.number_input(
                        "Pico giro 3 m (s)", value=float(sinais["G1_lat"]),
                        step=0.05, key=f"g1_{paciente}",
                    )
                with cc4:
                    novo_g2 = st.number_input(
                        "Pico giro 6 m (s)", value=float(sinais["G2_lat"]),
                        step=0.05, key=f"g2_{paciente}",
                    )
                if st.button("✅ Aplicar correção", key=f"corrigir_{paciente}"):
                    sinais_corrigido = dict(sinais)
                    sinais_corrigido["start_test"] = novo_inicio
                    sinais_corrigido["stop_test"] = novo_fim
                    sinais_corrigido["G1_lat"] = novo_g1
                    sinais_corrigido["G2_lat"] = novo_g2
                    sinais_corrigido["G1_amp"] = _y_no_instante(
                        sinais["t_novo_gyro"], sinais["norma_gyro_filtrado"], novo_g1)
                    sinais_corrigido["G2_amp"] = _y_no_instante(
                        sinais["t_novo_gyro"], sinais["norma_gyro_filtrado"], novo_g2)

                    info["sinais"] = sinais_corrigido
                    info["metrics"] = metricas_a_partir_de_sinais(sinais_corrigido)
                    info["corrigido"] = True
                    st.rerun()
    else:
        st.info("Clique em \"Processar todos os pacientes\" para calcular os resultados.")
else:
    st.info("Aguardando o envio dos arquivos.")
