import os, json
import numpy as np

def _to_1d_list(x):
    # aceita: escalar, ndarray, lista de ndarrays (1 por ângulo)
    if isinstance(x, list):
        if len(x) == 1 and isinstance(x[0], (np.ndarray, list, tuple)):
            x = x[0]
        elif all(isinstance(a, (np.ndarray, list, tuple)) for a in x):
            # se vier uma lista de blocos, concatena
            x = np.concatenate([np.asarray(a).reshape(-1) for a in x], axis=0)
        else:
            return [float(v) for v in x]
    x = np.asarray(x).reshape(-1)
    return [float(v) for v in x]

def _broadcast(y, length):
    # y escalar -> repete; vetor de 1 -> repete; vetor de mesmo tamanho -> ok
    arr = np.asarray(y).reshape(-1)
    if arr.size == 1:
        return [float(arr[0])]*length
    if arr.size == length:
        return [float(v) for v in arr]
    raise ValueError(f"tau_target tem tamanho {arr.size}; esperado 1 ou {length}.")

def _load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def salvar_torques(path_arquivo, musculo, angulo, torque_predito, torque_medido):
    """
    Salva:
      musculo -> angulo -> {torque_medido: [...], torque_predito: [...]}
    - angulo pode ser int/float
    - torque_predito pode ser np.array, ou lista com np.array (seu caso)
    - torque_medido pode ser escalar ou vetor; se escalar, replica
    """
    dados = _load_json(path_arquivo)

    pred = _to_1d_list(torque_predito)
    # medido: igual tamanho do predito (ou replica se escalar)
    if np.isscalar(torque_medido) or (hasattr(torque_medido, "__len__") and len(np.asarray(torque_medido).reshape(-1)) in (1, len(pred))):
        meas = _broadcast(torque_medido, len(pred))
    else:
        raise ValueError("Formato de torque_medido inválido (use escalar ou vetor do mesmo tamanho do predito).")

    musc = str(musculo)
    ang = str(float(angulo)) if isinstance(angulo, (float, np.floating)) else str(int(angulo))

    if musc not in dados:
        dados[musc] = {}
    dados[musc][ang] = {
        "torque_medido": meas,
        "torque_predito": pred
    }

    _save_json(path_arquivo, dados)
    print(f"Salvo {len(pred)} amostras em '{path_arquivo}' -> {musc}/{ang}")