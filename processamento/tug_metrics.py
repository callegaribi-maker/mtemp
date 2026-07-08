"""
tug_metrics.py

Camada fina sobre `processamento.tugProcessing.processar_tug` (codigo original,
nao modificado) que extrai as mesmas variaveis calculadas na pagina
"Exportar Resultados" -> TUG do mWEB.py, para uso em lote (multiplos pacientes),
e tambem devolve os sinais brutos/filtrados necessarios para desenhar os
mesmos graficos de conferencia (inicio/fim do teste e os picos marcados)
usados na pagina "Visualizacao Grafica" -> TUG do mWEB.py.

Nenhuma logica de processamento de sinal foi alterada em relacao ao app
original: as funcoes abaixo apenas repetem, para cada paciente, exatamente os
mesmos passos que o mWEB.py executa para um unico paciente.
"""

import numpy as np

from processamento import tugProcessing


def processar_paciente_tug(dados_acc, dados_gyro, baseline_onset, baseline_offset,
                            filtro_acc=2, filtro_gyro=1.25):
    """
    Roda o processamento TUG (identico ao mWEB.py) para um paciente e retorna
    um dicionario com:
      - "metrics": as mesmas metricas exibidas/exportadas pelo app original
      - "sinais": os vetores usados nos graficos de conferencia (inicio/fim
        do teste e localizacao dos picos), no mesmo formato do mWEB.py
    """
    (t_novo_acc, v_acc, ml_acc, z_acc_filtrado, norma_acc_filtrado,
     t_novo_gyro, v_gyro, ml_gyro, z_gyro_filtrado, norma_gyro_filtrado,
     start_test, stop_test, idx, idx_ml, idx_acc_ap, idx_acc_v,
     duration) = tugProcessing.processar_tug(
        dados_acc, dados_gyro, filtro_acc, filtro_gyro, baseline_onset, baseline_offset
    )

    vertical_squared = np.sqrt(v_gyro ** 2)
    lat1, lat2 = idx[1][0], idx[1][1]
    amp1, amp2 = vertical_squared[idx[0][0]], vertical_squared[idx[0][1]]
    if lat1 > lat2:
        G1_lat, G1_amp, G2_lat, G2_amp = lat2, amp2, lat1, amp1
    else:
        G1_lat, G1_amp, G2_lat, G2_amp = lat1, amp1, lat2, amp2

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

    metrics = {
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

    return {"metrics": metrics, "sinais": sinais}


def calcular_metricas_tug(dados_acc, dados_gyro, baseline_onset, baseline_offset,
                           filtro_acc=2, filtro_gyro=1.25):
    """Compatibilidade: retorna apenas o dicionario de metricas (sem os sinais)."""
    return processar_paciente_tug(
        dados_acc, dados_gyro, baseline_onset, baseline_offset, filtro_acc, filtro_gyro
    )["metrics"]
