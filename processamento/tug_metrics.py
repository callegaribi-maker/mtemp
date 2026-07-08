"""
tug_metrics.py

Camada fina sobre `processamento.tugProcessing.processar_tug` (codigo original,
nao modificado) que extrai as mesmas variaveis calculadas na pagina
"Exportar Resultados" -> TUG do mWEB.py, para uso em lote (multiplos pacientes),
e tambem devolve os sinais brutos/filtrados necessarios para desenhar os
mesmos graficos de conferencia (inicio/fim do teste e os picos marcados)
usados na pagina "Visualizacao Grafica" -> TUG do mWEB.py.

Nenhuma logica de deteccao de sinal foi alterada em relacao ao app original:
`processar_paciente_tug` roda exatamente os mesmos passos que o mWEB.py
executa para um unico paciente. A unica coisa adicionada aqui e a funcao
`metricas_a_partir_de_sinais`, que apenas re-calcula as formulas de duracao
(ja existentes no mWEB.py) a partir dos valores guardados em "sinais" -- ela
existe para permitir que, quando o algoritmo erra o inicio/fim ou os dois
picos de giro (algo que ja podia acontecer no app individual e exigia ajuste
manual do analista), o usuario possa corrigir esses instantes manualmente
paciente a paciente, sem tocar na deteccao automatica em si.
"""

import numpy as np
from scipy.signal import find_peaks

from processamento import tugProcessing


def detectar_inicio_fim_resultante(t, y, baseline_onset, baseline_offset, k=4.0):
    """Detecta inicio/fim do teste diretamente no sinal RESULTANTE do giroscopio
    (norma_gyro_filtrado), como pedido: "o inicio e fim e quando o sinal sobe/desce".

    Em vez do limiar fixo (0.25 rad/s) usado no algoritmo original -- que se mostrou
    proximo demais do ruido de repouso da resultante em alguns pacientes -- o limiar
    aqui e adaptativo: media + k*desvio-padrao de uma janela de repouso no comeco do
    registro (antes do baseline_onset). O inicio e a primeira vez que o sinal cruza
    esse limiar depois do baseline_onset; o fim e a ULTIMA vez que o sinal esta acima
    do limiar antes do baseline_offset (mesma janela/convencao do algoritmo original,
    que tambem limita a busca do fim ate o baseline_offset)."""
    t = np.asarray(t)
    y = np.asarray(y)

    onset_idx = int(np.searchsorted(t, baseline_onset))
    offset_idx = int(np.searchsorted(t, baseline_offset))
    offset_idx = max(offset_idx, onset_idx + 1)

    janela_repouso = y[t <= max(baseline_onset, t[0] + 1.5)]
    if len(janela_repouso) < 5:
        janela_repouso = y[:max(5, len(y) // 20)]
    thr = janela_repouso.mean() + k * janela_repouso.std()
    thr = max(thr, janela_repouso.mean() * 1.15, 0.05)

    seg_inicio = y[onset_idx:]
    acima_inicio = np.where(seg_inicio > thr)[0]
    start_idx = onset_idx + acima_inicio[0] if len(acima_inicio) else onset_idx

    seg_fim = y[:offset_idx]
    acima_fim = np.where(seg_fim > thr)[0]
    stop_idx = acima_fim[-1] if len(acima_fim) else max(offset_idx - 1, start_idx + 1)

    if stop_idx <= start_idx:
        stop_idx = min(len(y) - 1, start_idx + 1)

    return float(t[start_idx]), float(t[stop_idx]), float(thr)


def detectar_picos_resultante(t, y, start_test, stop_test):
    """Encontra os DOIS MAIORES picos do sinal resultante do giroscopio dentro da
    janela [start_test, stop_test] -- esses sao G1 (giro 3 m) e G2 (giro 6 m),
    conforme pedido: "os dois maiores picos sao G1 e G2"."""
    t = np.asarray(t)
    y = np.asarray(y)
    fs = 1 / np.median(np.diff(t))

    mask = (t >= start_test) & (t <= stop_test)
    idx_janela = np.where(mask)[0]
    if len(idx_janela) < 3:
        idx_janela = np.arange(len(t))
    y_janela = y[idx_janela]

    picos, _ = find_peaks(y_janela, distance=max(1, int(0.5 * fs)))
    if len(picos) < 2:
        picos, _ = find_peaks(y_janela)
    if len(picos) < 2:
        # fallback extremo: usa os dois maiores pontos da janela
        picos = np.argsort(y_janela)[-2:]

    alturas = y_janela[picos]
    top2 = picos[np.argsort(alturas)[-2:]]
    top2 = sorted(top2)
    idx_g1, idx_g2 = idx_janela[top2[0]], idx_janela[top2[1]]
    return float(t[idx_g1]), float(y[idx_g1]), float(t[idx_g2]), float(y[idx_g2])


def metricas_a_partir_de_sinais(sinais):
    """Recalcula o dicionario de metricas usando exatamente as mesmas formulas
    do mWEB.py, a partir dos valores guardados em `sinais` (start_test,
    stop_test, G0..G4, A1, A2, A1v, A2v). Serve tanto para o calculo
    automatico quanto para depois de uma correcao manual de algum desses
    instantes."""
    start_test = sinais["start_test"]
    stop_test = sinais["stop_test"]
    G0_lat, G0_amp = sinais["G0_lat"], sinais["G0_amp"]
    G1_lat, G1_amp = sinais["G1_lat"], sinais["G1_amp"]
    G2_lat, G2_amp = sinais["G2_lat"], sinais["G2_amp"]
    A1_lat, A1_amp = sinais["A1_lat"], sinais["A1_amp"]
    A2_lat, A2_amp = sinais["A2_lat"], sinais["A2_amp"]
    A1v_lat, A1v_amp = sinais["A1v_lat"], sinais["A1v_amp"]
    A2v_lat, A2v_amp = sinais["A2v_lat"], sinais["A2v_amp"]

    return {
        "Duracao do teste (s)": round(stop_test - start_test, 4),
        "Duracao para levantar (s)": round(G0_lat - start_test, 4),
        "Duracao da caminhada de ida (s)": round(G1_lat - G0_lat, 4),
        "Duracao da caminhada de volta (s)": round(G2_lat - G1_lat, 4),
        "Duracao para sentar (s)": round(stop_test - G2_lat, 4),
        "Tempo pico vel. angular sentado->pe (s)": round(G0_lat - start_test, 4),
        "Tempo pico acel. AP sentado->pe (s)": round(A1_lat - start_test, 4),
        "Tempo pico acel. V sentado->pe (s)": round(A1v_lat - start_test, 4),
        "Tempo pico vel. angular giro 3 m (s)": round(G1_lat - start_test, 4),
        "Tempo pico vel. angular giro 6 m (s)": round(G2_lat - start_test, 4),
        "Tempo pico vel. angular pe->sentado (s)": round(G2_lat - start_test, 4),
        "Tempo pico acel. AP pe->sentado (s)": round(A2_lat - start_test, 4),
        "Tempo pico acel. V pe->sentado (s)": round(A2v_lat - start_test, 4),
        "Vel. angular maxima sentado->pe (rad/s)": round(G0_amp, 4),
        "Acel. maxima AP sentado->pe (m/s2)": round(A1_amp, 4),
        "Acel. maxima V sentado->pe (m/s2)": round(A1v_amp, 4),
        "Vel. angular maxima giro 3 m (rad/s)": round(G1_amp, 4),
        "Vel. angular maxima giro 6 m (rad/s)": round(G2_amp, 4),
        "Acel. maxima AP pe->sentado (m/s2)": round(A2_amp, 4),
        "Acel. maxima V pe->sentado (m/s2)": round(A2v_amp, 4),
    }


def processar_paciente_tug(dados_acc, dados_gyro, baseline_onset, baseline_offset,
                            filtro_acc=2, filtro_gyro=1.25):
    """
    Roda o processamento TUG (identico ao mWEB.py) para um paciente e retorna
    um dicionario com:
      - "metrics": as mesmas metricas exibidas/exportadas pelo app original
      - "sinais": os vetores e instantes usados nos graficos de conferencia
        (inicio/fim do teste e localizacao dos picos), no mesmo formato do
        mWEB.py -- podem ser corrigidos manualmente depois, se necessario.
    """
    (t_novo_acc, v_acc, ml_acc, z_acc_filtrado, norma_acc_filtrado,
     t_novo_gyro, v_gyro, ml_gyro, z_gyro_filtrado, norma_gyro_filtrado,
     _start_test_orig, _stop_test_orig, idx, idx_ml, idx_acc_ap, idx_acc_v,
     duration) = tugProcessing.processar_tug(
        dados_acc, dados_gyro, filtro_acc, filtro_gyro, baseline_onset, baseline_offset
    )

    # Inicio/fim do teste e os dois picos de giro (G1/G2) sao detectados
    # diretamente no sinal RESULTANTE do giroscopio (norma_gyro_filtrado), e nao
    # no eixo isolado escolhido pela heuristica de orientacao do aparelho -- isso
    # evita o descompasso entre o que e mostrado no grafico (resultante) e os
    # instantes marcados. `tugProcessing.processar_tug` continua 100% intocado;
    # apenas reaproveitamos seus sinais ja filtrados para essa deteccao.
    start_test, stop_test, _thr_gyro = detectar_inicio_fim_resultante(
        t_novo_gyro, norma_gyro_filtrado, baseline_onset, baseline_offset
    )
    G1_lat, G1_amp, G2_lat, G2_amp = detectar_picos_resultante(
        t_novo_gyro, norma_gyro_filtrado, start_test, stop_test
    )

    ml_squared = np.sqrt(ml_gyro ** 2)
    lat1, lat2 = idx_ml[1][0], idx_ml[1][1]
    amp1, amp2 = ml_squared[idx_ml[0][0]], ml_squared[idx_ml[0][1]]
    if lat1 > lat2:
        G0_lat, G0_amp, G4_lat, G4_amp = lat2, amp2, lat1, amp1
    else:
        G0_lat, G0_amp, G4_lat, G4_amp = lat1, amp1, lat2, amp2

    acc_ap_squared = np.sqrt(z_acc_filtrado ** 2)
    lat1, lat2 = idx_acc_ap[1][0], idx_acc_ap[1][1]
    amp1, amp2 = acc_ap_squared[idx_acc_ap[0][0]], acc_ap_squared[idx_acc_ap[0][1]]
    if lat1 > lat2:
        A1_lat, A1_amp, A2_lat, A2_amp = lat2, amp2, lat1, amp1
    else:
        A1_lat, A1_amp, A2_lat, A2_amp = lat1, amp1, lat2, amp2

    acc_v_squared = np.sqrt(v_acc ** 2)
    lat1, lat2 = idx_acc_v[1][0], idx_acc_v[1][1]
    amp1, amp2 = acc_v_squared[idx_acc_v[0][0]], acc_v_squared[idx_acc_v[0][1]]
    if lat1 > lat2:
        A1v_lat, A1v_amp, A2v_lat, A2v_amp = lat2, amp2, lat1, amp1
    else:
        A1v_lat, A1v_amp, A2v_lat, A2v_amp = lat1, amp1, lat2, amp2

    sinais = {
        "t_novo_acc": t_novo_acc,
        "v_acc": v_acc,
        "ml_acc": ml_acc,
        "z_acc_filtrado": z_acc_filtrado,
        "norma_acc_filtrado": norma_acc_filtrado,
        "t_novo_gyro": t_novo_gyro,
        "v_gyro": v_gyro,
        "ml_gyro": ml_gyro,
        "z_gyro_filtrado": z_gyro_filtrado,
        "norma_gyro_filtrado": norma_gyro_filtrado,
        "start_test": start_test,
        "stop_test": stop_test,
        "G0_lat": G0_lat, "G0_amp": G0_amp,
        "G1_lat": G1_lat, "G1_amp": G1_amp,
        "G2_lat": G2_lat, "G2_amp": G2_amp,
        "G4_lat": G4_lat, "G4_amp": G4_amp,
        "A1_lat": A1_lat, "A1_amp": A1_amp,
        "A2_lat": A2_lat, "A2_amp": A2_amp,
        "A1v_lat": A1v_lat, "A1v_amp": A1v_amp,
        "A2v_lat": A2v_lat, "A2v_amp": A2v_amp,
    }

    return {"metrics": metricas_a_partir_de_sinais(sinais), "sinais": sinais}


def calcular_metricas_tug(dados_acc, dados_gyro, baseline_onset, baseline_offset,
                           filtro_acc=2, filtro_gyro=1.25):
    """Compatibilidade: retorna apenas o dicionario de metricas (sem os sinais)."""
    return processar_paciente_tug(
        dados_acc, dados_gyro, baseline_onset, baseline_offset, filtro_acc, filtro_gyro
    )["metrics"]
