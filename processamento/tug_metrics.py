"""
tug_metrics.py

Camada fina sobre `processamento.tugProcessing.processar_tug` (codigo original,
nao modificado) que reaproveita os sinais RESULTANTES (norma) ja filtrados por
ele -- resultante do acelerometro (norma_acc_filtrado) e do giroscopio
(norma_gyro_filtrado) -- para detectar todos os marcos do TUG (inicio, fim,
pico sentado->pe, os dois picos de giro dos giros e os dois picos de
aceleracao das transicoes), em vez de usar eixos isolados escolhidos por uma
heuristica de orientacao do aparelho.

`tugProcessing.processar_tug` continua 100% intocado. `metricas_a_partir_de_sinais`
recalcula as metricas (nomes/ordem iguais aos de uma tabela de referencia
publicada) a partir dos valores guardados em "sinais" -- permite corrigir
manualmente qualquer um desses instantes, paciente a paciente, sem alterar a
deteccao automatica em si.
"""

import numpy as np
from scipy.signal import find_peaks

from processamento import tugProcessing


def detectar_inicio_fim_resultante(t, y, baseline_onset, baseline_offset, k=4.0):
    """Detecta inicio/fim do teste diretamente no sinal RESULTANTE do giroscopio
    (norma_gyro_filtrado): "o inicio e fim e quando o sinal sobe/desce".

    O limiar e adaptativo: media + k*desvio-padrao de uma janela de repouso no
    comeco do registro (antes do baseline_onset) -- evita depender de um valor
    fixo (0.25 rad/s) que pode ficar perto demais do ruido de repouso da
    resultante em alguns pacientes. O inicio e a primeira vez que o sinal cruza
    esse limiar depois do baseline_onset; o fim e a ULTIMA vez que o sinal esta
    acima do limiar antes do baseline_offset (mesma janela/convencao do
    algoritmo original, que tambem limita a busca do fim ate o baseline_offset)."""
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


def detectar_dois_maiores_picos(t, y, tmin, tmax):
    """Encontra os DOIS MAIORES picos do sinal resultante do giroscopio dentro
    da janela [tmin, tmax] -- G1 (giro 3 m) e G2 (giro 6 m): "os dois maiores
    picos sao G1 e G2"."""
    t = np.asarray(t)
    y = np.asarray(y)
    fs = 1 / np.median(np.diff(t))

    mask = (t >= tmin) & (t <= tmax)
    idx_janela = np.where(mask)[0]
    if len(idx_janela) < 3:
        idx_janela = np.arange(len(t))
    y_janela = y[idx_janela]

    picos, _ = find_peaks(y_janela, distance=max(1, int(0.5 * fs)))
    if len(picos) < 2:
        picos, _ = find_peaks(y_janela)
    if len(picos) < 2:
        picos = np.argsort(y_janela)[-2:]

    alturas = y_janela[picos]
    top2 = picos[np.argsort(alturas)[-2:]]
    top2 = sorted(top2)
    idx_1, idx_2 = idx_janela[top2[0]], idx_janela[top2[1]]
    return float(t[idx_1]), float(y[idx_1]), float(t[idx_2]), float(y[idx_2])


def detectar_maior_pico_local(t, y, tmin, tmax):
    """Encontra o maior PICO LOCAL (maximo local de verdade, via find_peaks --
    nao apenas o maior valor bruto da janela, que poderia "roubar" o valor
    logo na borda da janela) dentro de [tmin, tmax]. Usado para achar o pico
    de giro sentado->pe (G0), que costuma ser bem menor que os picos dos
    giros (G1/G2) mas ainda assim um maximo local genuino logo apos o inicio
    do teste."""
    t = np.asarray(t)
    y = np.asarray(y)
    mask = (t >= tmin) & (t <= tmax)
    idx_janela = np.where(mask)[0]
    if len(idx_janela) < 3:
        if len(idx_janela) == 0:
            return float(tmin), float(y[int(np.searchsorted(t, tmin))])
        melhor = idx_janela[np.argmax(y[idx_janela])]
        return float(t[melhor]), float(y[melhor])

    y_janela = y[idx_janela]
    picos, _ = find_peaks(y_janela)
    if len(picos) == 0:
        melhor = idx_janela[np.argmax(y_janela)]
    else:
        melhor = idx_janela[picos[np.argmax(y_janela[picos])]]
    return float(t[melhor]), float(y[melhor])


def maior_valor_no_intervalo(t, y, tmin, tmax):
    """Maior valor bruto do sinal resultante do acelerometro dentro de
    [tmin, tmax] -- usado para os picos de aceleracao das transicoes
    (sentado->pe e pe->sentado), que ja ficam bem delimitados pela janela
    (nao precisam do cuidado de "pico local" feito para G0)."""
    t = np.asarray(t)
    y = np.asarray(y)
    mask = (t >= tmin) & (t <= tmax)
    idx_janela = np.where(mask)[0]
    if len(idx_janela) == 0:
        idx = int(np.argmin(np.abs(t - tmin)))
        return float(t[idx]), float(y[idx])
    melhor = idx_janela[np.argmax(y[idx_janela])]
    return float(t[melhor]), float(y[melhor])


def metricas_a_partir_de_sinais(sinais):
    """Recalcula o dicionario de metricas a partir dos valores guardados em
    `sinais` (start_test, stop_test, G0/G1/G2, A1/A2). Nomes e ordem seguem
    uma tabela de referencia publicada (fases do TUG + picos de aceleracao e
    velocidade angular). Serve tanto para o calculo automatico quanto para
    depois de uma correcao manual de qualquer um desses instantes."""
    start_test = sinais["start_test"]
    stop_test = sinais["stop_test"]
    G0_lat = sinais["G0_lat"]
    G1_lat, G1_amp = sinais["G1_lat"], sinais["G1_amp"]
    G2_lat, G2_amp = sinais["G2_lat"], sinais["G2_amp"]
    A1_amp = sinais["A1_amp"]
    A2_amp = sinais["A2_amp"]

    return {
        "Total duration (s)": round(stop_test - start_test, 4),
        "Sit-to-Stand transition duration (s)": round(G0_lat - start_test, 4),
        "Forward walking duration (s)": round(G1_lat - G0_lat, 4),
        "Return walking duration (s)": round(G2_lat - G1_lat, 4),
        "Stand-to-Sit transition duration (s)": round(stop_test - G2_lat, 4),
        "Peak acceleration during Sit-to-Stand duration (m/s²)": round(A1_amp, 4),
        "Peak acceleration during Stand-to-Sit duration (m/s²)": round(A2_amp, 4),
        "Peak angular velocity during 3-meters turning (rad/s)": round(G1_amp, 4),
        "Peak angular velocity during the turn in front of the chair (rad/s)": round(G2_amp, 4),
    }


def processar_paciente_tug(dados_acc, dados_gyro, baseline_onset, baseline_offset,
                            filtro_acc=2, filtro_gyro=1.25):
    """
    Roda o processamento TUG (identico ao mWEB.py, via tugProcessing.processar_tug,
    sem alteracao) para um paciente e retorna um dicionario com:
      - "metrics": as metricas no formato da tabela de referencia (ver
        `metricas_a_partir_de_sinais`)
      - "sinais": os vetores e instantes usados nos graficos de conferencia
        (inicio/fim do teste e localizacao dos picos) -- podem ser corrigidos
        manualmente depois, se necessario.

    Todos os marcos (inicio, fim, G0, G1, G2, e os picos de aceleracao A1/A2)
    sao detectados nos sinais RESULTANTES (norma_gyro_filtrado / norma_acc_filtrado)
    devolvidos por tugProcessing.processar_tug -- o mesmo sinal mostrado nos
    graficos de conferencia.
    """
    (t_novo_acc, v_acc, ml_acc, z_acc_filtrado, norma_acc_filtrado,
     t_novo_gyro, v_gyro, ml_gyro, z_gyro_filtrado, norma_gyro_filtrado,
     _start_test_orig, _stop_test_orig, _idx, _idx_ml, _idx_acc_ap, _idx_acc_v,
     _duration) = tugProcessing.processar_tug(
        dados_acc, dados_gyro, filtro_acc, filtro_gyro, baseline_onset, baseline_offset
    )

    start_test, stop_test, _thr_gyro = detectar_inicio_fim_resultante(
        t_novo_gyro, norma_gyro_filtrado, baseline_onset, baseline_offset
    )
    G1_lat, G1_amp, G2_lat, G2_amp = detectar_dois_maiores_picos(
        t_novo_gyro, norma_gyro_filtrado, start_test, stop_test
    )
    G0_lat, G0_amp = detectar_maior_pico_local(
        t_novo_gyro, norma_gyro_filtrado, start_test, G1_lat
    )
    A1_lat, A1_amp = maior_valor_no_intervalo(
        t_novo_acc, norma_acc_filtrado, start_test, G0_lat
    )
    A2_lat, A2_amp = maior_valor_no_intervalo(
        t_novo_acc, norma_acc_filtrado, G2_lat, stop_test
    )

    sinais = {
        "t_novo_acc": t_novo_acc,
        "norma_acc_filtrado": norma_acc_filtrado,
        "t_novo_gyro": t_novo_gyro,
        "norma_gyro_filtrado": norma_gyro_filtrado,
        "start_test": start_test,
        "stop_test": stop_test,
        "G0_lat": G0_lat, "G0_amp": G0_amp,
        "G1_lat": G1_lat, "G1_amp": G1_amp,
        "G2_lat": G2_lat, "G2_amp": G2_amp,
        "A1_lat": A1_lat, "A1_amp": A1_amp,
        "A2_lat": A2_lat, "A2_amp": A2_amp,
    }

    return {"metrics": metricas_a_partir_de_sinais(sinais), "sinais": sinais}


def recalcular_picos_acc(sinais):
    """Recalcula A1/A2 (picos de aceleracao resultante) a partir das janelas
    [start_test, G0_lat] e [G2_lat, stop_test] em `sinais` -- chamar depois de
    qualquer correcao manual de start/G0/G2/stop, para que os picos de
    aceleracao acompanhem as novas janelas em vez de ficarem com o valor
    antigo."""
    sinais = dict(sinais)
    A1_lat, A1_amp = maior_valor_no_intervalo(
        sinais["t_novo_acc"], sinais["norma_acc_filtrado"],
        sinais["start_test"], sinais["G0_lat"],
    )
    A2_lat, A2_amp = maior_valor_no_intervalo(
        sinais["t_novo_acc"], sinais["norma_acc_filtrado"],
        sinais["G2_lat"], sinais["stop_test"],
    )
    sinais["A1_lat"], sinais["A1_amp"] = A1_lat, A1_amp
    sinais["A2_lat"], sinais["A2_amp"] = A2_lat, A2_amp
    return sinais


def calcular_metricas_tug(dados_acc, dados_gyro, baseline_onset, baseline_offset,
                           filtro_acc=2, filtro_gyro=1.25):
    """Compatibilidade: retorna apenas o dicionario de metricas (sem os sinais)."""
    return processar_paciente_tug(
        dados_acc, dados_gyro, baseline_onset, baseline_offset, filtro_acc, filtro_gyro
    )["metrics"]
