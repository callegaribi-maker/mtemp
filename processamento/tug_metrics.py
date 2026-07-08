"""
tug_metrics.py

Camada fina sobre `processamento.tugProcessing.processar_tug` (código original,
não modificado) que extrai as mesmas variáveis calculadas na página
"Exportar Resultados" -> TUG do mWEB.py, para uso em lote (múltiplos pacientes).

Nenhuma lógica de processamento de sinal foi alterada em relação ao app
original: esta função apenas repete, para cada paciente, exatamente os
mesmos passos que o mWEB.py executa para um único paciente.
"""

import numpy as np

from processamento import tugProcessing


def calcular_metricas_tug(dados_acc, dados_gyro, baseline_onset, baseline_offset,
                           filtro_acc=2, filtro_gyro=1.25):
    """
    Roda o processamento TUG (idêntico ao mWEB.py) para um paciente e retorna
    um dicionário com as mesmas métricas exibidas/exportadas pelo app original.
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

    return {
        "Duração do teste (s)": round(stop_test - start_test, 4),
        "Duração para levantar (s)": round(G0_lat - start_test, 4),
        "Duração da caminhada de ida (s)": round(G1_lat - G0_lat, 4),
        "Duração da caminhada de volta (s)": round(G2_lat - G1_lat, 4),
        "Duração para sentar (s)": round(stop_test - G2_lat, 4),
        "Tempo pico vel. angular sentado->pé (s)": round(G0_lat - start_test, 4),
        "Tempo pico acel. AP sentado->pé (s)": round(A1_lat - start_test, 4),
        "Tempo pico acel. V sentado->pé (s)": round(A1v_lat - start_test, 4),
        "Tempo pico vel. angular giro 3 m (s)": round(G1_lat - start_test, 4),
        "Tempo pico vel. angular giro 6 m (s)": round(G2_lat - start_test, 4),
        "Tempo pico vel. angular pé->sentado (s)": round(G2_lat - start_test, 4),
        "Tempo pico acel. AP pé->sentado (s)": round(A2_lat - start_test, 4),
        "Tempo pico acel. V pé->sentado (s)": round(A2v_lat - start_test, 4),
        "Vel. angular máxima sentado->pé (rad/s)": round(G0_amp, 4),
        "Acel. máxima AP sentado->pé (m/s2)": round(A1_amp, 4),
        "Acel. máxima V sentado->pé (m/s2)": round(A1v_amp, 4),
        "Vel. angular máxima giro 3 m (rad/s)": round(G1_amp, 4),
        "Vel. angular máxima giro 6 m (rad/s)": round(G2_amp, 4),
        "Acel. máxima AP pé->sentado (m/s2)": round(A2_amp, 4),
        "Acel. máxima V pé->sentado (m/s2)": round(A2v_amp, 4),
    }
