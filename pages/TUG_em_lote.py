"""
TUG em lote (múltiplos pacientes)
==================================

Página adicional do Momentum Web, criada para permitir a análise do TUG de
vários pacientes de uma só vez, sem alterar em nada o app original
(`mWEB.py`) nem o módulo de processamento (`processamento/tugProcessing.py`).

Como funciona
-------------
1. Envie, de uma só vez, os arquivos de acelerômetro e giroscópio de todos os
   pacientes (o par de cada paciente é identificado automaticamente pelo
   nome do arquivo, no padrão usado pelo app: "<Paciente> tug_acelData_...txt"
   e "<Paciente> tug_gyroData_...txt").
2. Para cada paciente, roda-se exatamente o mesmo processamento usado na
   análise individual (mesmos filtros, mesma lógica de início/fim do teste).
3. O resultado sai em bloco: uma tabela com uma linha por paciente e as
   mesmas variáveis que o app exporta no modo individual, pronta para
   baixar em CSV/TXT.

Esta página aparece automaticamente no menu lateral do Streamlit (pasta
`pages/`) quando o app é implantado a partir deste fork — o arquivo
`mWEB.py` continua sendo o ponto de entrada e não precisa de nenhuma
alteração.
"""

import io
import re

import numpy as np
import pandas as pd
import streamlit as st

from processamento import tugProcessing
from processamento.tug_metrics import calcular_metricas_tug

st.set_page_config(page_title="TUG em lote - Momentum Web", page_icon="🧍", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🧍 TUG — Análise em lote</h1>",
            unsafe_allow_html=True)
st.caption(
    "Envie os arquivos de acelerômetro e giroscópio de vários pacientes de uma vez. "
    "O pareamento acc/gyro de cada paciente é feito automaticamente pelo nome do arquivo."
)

AUDIO_EXTS = {"3ga", "aac", "m4a", "mp3", "wav", "ogg", "flac"}


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


def identificar_paciente(nome_arquivo: str) -> tuple[str, str]:
    """Extrai (nome_do_paciente, tipo) a partir do padrão de nome de arquivo
    usado pelo Momentum ('<Paciente> tug_acelData_...' / '... tug_gyroData_...').
    Retorna tipo = 'acel', 'gyro' ou None se não reconhecido."""
    m = re.search(r"(.*?)\s*tug_(acel|gyro)Data", nome_arquivo, re.IGNORECASE)
    if not m:
        return nome_arquivo, None
    paciente = m.group(1).strip()
    tipo = "acel" if m.group(2).lower() == "acel" else "gyro"
    return paciente, tipo


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
            erros = []
            barra = st.progress(0.0)
            for i, (paciente, arqs) in enumerate(sorted(completos.items())):
                try:
                    dados_acc = carregar_dados_generico(arqs["acel"].name, arqs["acel"].getvalue())
                    dados_gyro = carregar_dados_generico(arqs["gyro"].name, arqs["gyro"].getvalue())

                    baseline_offset = (np.max(dados_acc["Tempo"]) / 1000) - margem_final

                    metrics = calcular_metricas_tug(
                        dados_acc, dados_gyro,
                        baseline_onset_global, baseline_offset,
                        filtro_acc=filtro_acc, filtro_gyro=filtro_gyro,
                    )

                    # sinaliza casos que provavelmente precisam de checagem manual
                    # (mesma checagem que um analista faria olhando os gráficos
                    # no modo individual: fases não podem ter duração negativa)
                    fases = [
                        metrics["Duração para levantar (s)"],
                        metrics["Duração da caminhada de ida (s)"],
                        metrics["Duração da caminhada de volta (s)"],
                        metrics["Duração para sentar (s)"],
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
                resultado = pd.DataFrame(linhas)
                st.subheader("3) Resultado em bloco")
                if (resultado["Alerta"] != "").any():
                    st.info(
                        "Pacientes marcados com alerta tiveram alguma fase do TUG com duração "
                        "negativa — isso indica que os picos identificados provavelmente não "
                        "correspondem às transições reais do teste. Revise esses casos na análise "
                        "individual (ajustando a janela de baseline) antes de usar os valores."
                    )
                st.dataframe(resultado, use_container_width=True)

                csv_bytes = resultado.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📄 Baixar resultados em lote (.csv)",
                    data=csv_bytes,
                    file_name="resultados_TUG_lote.csv",
                    mime="text/csv",
                )

                txt_bytes = resultado.to_csv(index=False, sep="\t").encode("utf-8")
                st.download_button(
                    "📄 Baixar resultados em lote (.txt, tab-separado)",
                    data=txt_bytes,
                    file_name="resultados_TUG_lote.txt",
                    mime="text/plain",
                )
    else:
        st.info("Envie ao menos um par completo (acelerômetro + giroscópio) de um paciente.")
else:
    st.info("Aguardando o envio dos arquivos.")
