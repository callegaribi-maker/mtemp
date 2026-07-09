"""
TUG em lote (multiplos pacientes) — versao standalone
========================================================

Este arquivo roda o mesmo processamento de TUG do mWEB.py (via
`processamento/tugProcessing.py`, sem nenhuma alteracao), mas permite
processar varios pacientes de uma vez e conferir/corrigir cada um.

Deteccao 100% a partir das curvas RESULTANTES
------------------------------------------------
Todos os marcos do teste sao detectados diretamente nos sinais RESULTANTES
(norma) ja filtrados por `tugProcessing.processar_tug` — os mesmos que
aparecem nos graficos:
  - Inicio/fim do teste: onde a resultante do giroscopio sobe/desce acima de
    um limiar adaptativo (calculado a partir do ruido de repouso no comeco
    do registro de cada paciente).
  - G0 (pico sentado→pé): maior pico local da resultante do giroscopio entre
    o inicio e o giro de 3 m.
  - G1/G2 (os dois giros): os dois maiores picos da resultante do giroscopio
    dentro da janela do teste.
  - Picos de aceleração (sentado→pé e pé→sentado): maior valor da resultante
    do acelerômetro dentro da janela de cada transição.
`tugProcessing.processar_tug` continua 100% intocado — essa deteccao roda em
cima dos sinais ja filtrados que ele devolve (ver `processamento/tug_metrics.py`).

Por que existe correcao manual por paciente
---------------------------------------------
Mesmo com o limiar adaptativo, o ruido de repouso no FINAL do registro pode
variar de paciente para paciente e a deteccao automatica pode não ser
perfeita em 100% dos casos. Por isso cada paciente tem campos para corrigir
manualmente início, G0, G1, G2 e fim — os picos de aceleração são
recalculados automaticamente a partir das novas janelas.
"""

import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from processamento.tug_metrics import (
    processar_paciente_tug,
    metricas_a_partir_de_sinais,
    recalcular_picos_acc,
)

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
    return float(y_array[idx])


def _metrics_para_linha(paciente, metrics, corrigido):
    fases = [
        metrics["Sit-to-Stand transition duration (s)"],
        metrics["Forward walking duration (s)"],
        metrics["Return walking duration (s)"],
        metrics["Stand-to-Sit transition duration (s)"],
    ]
    alerta = "Revisar (fase com duração negativa)" if any(f < 0 for f in fases) else ""
    return {"Paciente": paciente, **metrics, "Corrigido manualmente": "Sim" if corrigido else "",
            "Alerta": alerta}


def plotar_resultantes(sinais, paciente):
    t_acc = sinais["t_novo_acc"]
    norma_acc = sinais["norma_acc_filtrado"]
    t_gyro = sinais["t_novo_gyro"]
    norma_gyro = sinais["norma_gyro_filtrado"]
    start_test = sinais["start_test"]
    stop_test = sinais["stop_test"]

    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=t_acc, y=norma_acc, mode="lines", name="Aceleração resultante",
            line=dict(color="black", width=1),
            hovertemplate="t=%{x:.2f}s<br>%{y:.2f} m/s²<extra></extra>",
        ))
        fig1.add_vline(x=start_test, line=dict(color="green", dash="dash", width=1.5))
        fig1.add_vline(x=stop_test, line=dict(color="red", dash="dash", width=1.5))
        fig1.add_trace(go.Scatter(
            x=[sinais["A1_lat"]], y=[sinais["A1_amp"]], mode="markers",
            marker=dict(color="green", size=11), name="A1 (sentado→pé)",
            hovertemplate="A1: t=%{x:.2f}s<br>%{y:.2f} m/s²<extra></extra>",
        ))
        fig1.add_trace(go.Scatter(
            x=[sinais["A2_lat"]], y=[sinais["A2_amp"]], mode="markers",
            marker=dict(color="magenta", size=11), name="A2 (pé→sentado)",
            hovertemplate="A2: t=%{x:.2f}s<br>%{y:.2f} m/s²<extra></extra>",
        ))
        fig1.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Tempo (s)", yaxis_title="Aceleração resultante (m/s²)",
            hovermode="x unified", legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig1, use_container_width=True, key=f"plot_acc_{paciente}")

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=t_gyro, y=norma_gyro, mode="lines", name="Vel. angular resultante",
            line=dict(color="black", width=1),
            hovertemplate="t=%{x:.2f}s<br>%{y:.2f} rad/s<extra></extra>",
        ))
        fig2.add_vline(x=start_test, line=dict(color="green", dash="dash", width=1.5))
        fig2.add_vline(x=stop_test, line=dict(color="red", dash="dash", width=1.5))
        fig2.add_trace(go.Scatter(
            x=[sinais["G0_lat"]], y=[sinais["G0_amp"]], mode="markers",
            marker=dict(color="seagreen", size=11), name="G0 (sentado→pé)",
            hovertemplate="G0: t=%{x:.2f}s<br>%{y:.2f} rad/s<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=[sinais["G1_lat"]], y=[sinais["G1_amp"]], mode="markers",
            marker=dict(color="royalblue", size=11), name="G1 (giro 3 m)",
            hovertemplate="G1: t=%{x:.2f}s<br>%{y:.2f} rad/s<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=[sinais["G2_lat"]], y=[sinais["G2_amp"]], mode="markers",
            marker=dict(color="magenta", size=11), name="G2 (giro na cadeira)",
            hovertemplate="G2: t=%{x:.2f}s<br>%{y:.2f} rad/s<extra></extra>",
        ))
        fig2.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Tempo (s)", yaxis_title="Vel. angular resultante (rad/s)",
            hovermode="x unified", legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"plot_gyro_{paciente}")


def _aplicar_correcao(paciente):
    """Callback do submit do formulario de correcao. Formularios do Streamlit
    (`st.form`) so disparam um rerun quando o botao de submit e clicado, e
    nesse rerun TODOS os campos do formulario ja estao com os valores mais
    recentes digitados pelo usuario (mesmo que ele nao tenha saido do campo) —
    e o padrao mais robusto para "varios campos + um botao aplicar", evitando
    qualquer problema de o clique no botao "nao pegar" o valor recem-digitado."""
    info = st.session_state["tug_lote_dados"][paciente]
    sinais = info["sinais"]

    novo_inicio = st.session_state[f"start_{paciente}"]
    novo_g0 = st.session_state[f"g0_{paciente}"]
    novo_g1 = st.session_state[f"g1_{paciente}"]
    novo_g2 = st.session_state[f"g2_{paciente}"]
    novo_fim = st.session_state[f"stop_{paciente}"]

    sinais_corrigido = dict(sinais)
    sinais_corrigido["start_test"] = novo_inicio
    sinais_corrigido["stop_test"] = novo_fim
    sinais_corrigido["G0_lat"] = novo_g0
    sinais_corrigido["G1_lat"] = novo_g1
    sinais_corrigido["G2_lat"] = novo_g2
    sinais_corrigido["G0_amp"] = _y_no_instante(
        sinais["t_novo_gyro"], sinais["norma_gyro_filtrado"], novo_g0)
    sinais_corrigido["G1_amp"] = _y_no_instante(
        sinais["t_novo_gyro"], sinais["norma_gyro_filtrado"], novo_g1)
    sinais_corrigido["G2_amp"] = _y_no_instante(
        sinais["t_novo_gyro"], sinais["norma_gyro_filtrado"], novo_g2)
    # Os picos de aceleração (A1/A2) são o maior valor dentro das janelas
    # [início, G0] e [G2, fim] — como essas janelas podem ter mudado, eles
    # são recalculados aqui em vez de manter o valor antigo.
    sinais_corrigido = recalcular_picos_acc(sinais_corrigido)

    info["sinais"] = sinais_corrigido
    info["metrics"] = metricas_a_partir_de_sinais(sinais_corrigido)
    info["corrigido"] = True
    st.session_state["_ultima_correcao"] = paciente


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
if "conferencia_concluida" not in st.session_state:
    st.session_state["conferencia_concluida"] = False

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
            st.session_state["conferencia_concluida"] = False
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
        if st.session_state.get("_ultima_correcao"):
            st.success(f"Correção aplicada para {st.session_state['_ultima_correcao']}.")
            st.session_state["_ultima_correcao"] = None

        st.subheader("3) Conferir e corrigir cada paciente")
        st.caption(
            "Passe o mouse sobre os gráficos para ver o tempo (eixo x) e o valor exato em "
            "cada ponto. Verde/vermelho = início e fim do teste. Os pontos marcados são G0 "
            "(sentado→pé), G1 (giro de 3 m), G2 (giro na frente da cadeira) no gráfico do "
            "giroscópio, e A1/A2 (picos de aceleração) no gráfico do acelerômetro. Se algum "
            "instante não estiver no lugar certo, digite o valor correto (em segundos) e "
            "clique em \"Aplicar correção\" — as métricas do paciente são recalculadas na "
            "hora, sem mexer na detecção automática dos demais pacientes. O resultado em "
            "bloco só aparece depois que você concluir a conferência, no final da página."
        )
        for paciente in sorted(dados_processados.keys()):
            info = dados_processados[paciente]
            metrics = info["metrics"]
            fases = [
                metrics["Sit-to-Stand transition duration (s)"],
                metrics["Forward walking duration (s)"],
                metrics["Return walking duration (s)"],
                metrics["Stand-to-Sit transition duration (s)"],
            ]
            alerta_paciente = any(f < 0 for f in fases)
            titulo = f"👤 {paciente}" + (" ⚠️" if alerta_paciente else "") + \
                     (" ✅ corrigido" if info.get("corrigido") else "")
            with st.expander(titulo):
                plotar_resultantes(info["sinais"], paciente)

                st.markdown("**Corrigir manualmente (opcional)**")
                sinais = info["sinais"]
                # st.form: os campos só são "lidos" quando o botão de submit do
                # PRÓPRIO formulário é clicado, todos juntos, no mesmo instante —
                # isso evita qualquer situação em que o valor recém-digitado não
                # seja considerado ao clicar em aplicar.
                with st.form(key=f"form_correcao_{paciente}"):
                    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
                    with cc1:
                        st.number_input(
                            "Início (s)", value=float(sinais["start_test"]),
                            step=0.05, key=f"start_{paciente}",
                        )
                    with cc2:
                        st.number_input(
                            "G0 sentado→pé (s)", value=float(sinais["G0_lat"]),
                            step=0.05, key=f"g0_{paciente}",
                        )
                    with cc3:
                        st.number_input(
                            "G1 giro 3 m (s)", value=float(sinais["G1_lat"]),
                            step=0.05, key=f"g1_{paciente}",
                        )
                    with cc4:
                        st.number_input(
                            "G2 giro cadeira (s)", value=float(sinais["G2_lat"]),
                            step=0.05, key=f"g2_{paciente}",
                        )
                    with cc5:
                        st.number_input(
                            "Fim (s)", value=float(sinais["stop_test"]),
                            step=0.05, key=f"stop_{paciente}",
                        )
                    st.form_submit_button(
                        "✅ Aplicar correção",
                        on_click=_aplicar_correcao, args=(paciente,),
                    )

        st.divider()
        if not dados_processados:
            pass
        elif st.session_state.get("conferencia_concluida"):
            st.success("Conferência concluída — resultado em bloco liberado abaixo.")
            if st.button("↩️ Voltar a corrigir (esconder resultado em bloco)"):
                st.session_state["conferencia_concluida"] = False
                st.rerun()
        else:
            st.info(
                "Confira/corrija os pacientes acima. Quando terminar, clique no "
                "botão abaixo para liberar o resultado em bloco."
            )
            if st.button("✅ Concluir conferência e gerar resultado em bloco", type="primary"):
                st.session_state["conferencia_concluida"] = True
                st.rerun()

        if dados_processados and st.session_state.get("conferencia_concluida"):
            st.subheader("4) Resultado em bloco")
            linhas = [
                _metrics_para_linha(p, v["metrics"], v.get("corrigido", False))
                for p, v in sorted(dados_processados.items())
            ]
            resultado_df = pd.DataFrame(linhas)

            if (resultado_df["Alerta"] != "").any():
                st.info(
                    "Pacientes marcados com alerta tiveram alguma fase do TUG com duração "
                    "negativa — sinal de que algum dos marcos não foi bem identificado "
                    "automaticamente. Volte à seção 3 para corrigir."
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
    else:
        st.info("Clique em \"Processar todos os pacientes\" para calcular os resultados.")
else:
    st.info("Aguardando o envio dos arquivos.")
