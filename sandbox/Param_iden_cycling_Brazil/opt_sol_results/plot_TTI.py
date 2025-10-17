import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

# ==== config ====
JSON_ARQ = "Param_iden_cycling_Brazil/opt_sol_results/pickles/torques_crankset.json"   # caminho do JSON salvo
# Passo de tempo:
DT = 1/35  # constante (ex.: 0.01)
# ou, se cada ângulo tem dt distinto, use um dict:
# DT = {"60": 0.01, "88": 0.01, "100": 0.0083333}

# ==== helpers ====
def angle_key_for(dt_spec, ang_key):
    if not isinstance(dt_spec, dict):
        return None
    # tenta "60" / "60.0" de forma flexível
    if ang_key in dt_spec:
        return ang_key
    if ang_key.endswith(".0") and ang_key[:-2] in dt_spec:
        return ang_key[:-2]
    if ang_key + ".0" in dt_spec:
        return ang_key + ".0"
    # também tenta forma inteira
    try:
        as_int = str(int(float(ang_key)))
        if as_int in dt_spec:
            return as_int
    except Exception:
        pass
    return None

def get_dt(dt_spec, ang_key):
    if isinstance(dt_spec, dict):
        k = angle_key_for(dt_spec, ang_key)
        if k is None:
            raise KeyError(f"dt não definido para ângulo {ang_key}. Chaves disponíveis: {list(dt_spec)}")
        return float(dt_spec[k])
    return float(dt_spec)

def tti(y, dt):
    y = np.asarray(y, dtype=float).reshape(-1)
    return float(np.trapz(y, dx=dt))

# ==== carregar JSON e gerar listas ====
data = json.loads(Path(JSON_ARQ).read_text(encoding="utf-8"))

# ==== montar listas a partir do JSON ====
integrals_msrd_list = []
integrals_pdct_list = []
angles_list = []      # para cores (ângulos)
subjects_list = []    # para marcadores (músculos)

for muscle, angles in data.items():
    # todo: verify how to use only preselected crank angles to include to the plot
    for ang_key, vals in angles.items():
        dt = get_dt(DT, ang_key)                 # seu dt (float ou por ângulo)
        tti_meas = tti(vals["torque_medido"], dt)
        tti_pred = tti(vals["torque_predito"], dt)

        integrals_msrd_list.append(tti_meas)     # Y
        integrals_pdct_list.append(tti_pred)     # X
        angles_list.append(int(float(ang_key)))
        subjects_list.append(muscle)

# ==== regressão e métricas globais (predicted->X, measured->Y) ====
can_fit = len(integrals_pdct_list) >= 2 and (len(set(integrals_pdct_list)) > 1)
if can_fit:
    slope, intercept, r_value, p_value, std_err = linregress(integrals_pdct_list, integrals_msrd_list)
    r_squared = r_value**2
    rmse = np.sqrt(mean_squared_error(integrals_pdct_list, integrals_msrd_list))
else:
    slope = intercept = r_squared = rmse = np.nan

# ==== plot ====
plt.figure(figsize=(10, 6))

# converte listas para arrays numéricos
all_x = np.asarray(integrals_pdct_list, dtype=float)
all_y = np.asarray(integrals_msrd_list, dtype=float)
all_z = np.asarray(angles_list, dtype=float)  # ângulo para coloração
unique_subjects = sorted(set(subjects_list))

# define marcador por músculo
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
subject_marker_map = {subject: markers[i % len(markers)] for i, subject in enumerate(unique_subjects)}

# scatter — um ponto por combinação (cor pelo ângulo)
sc = plt.scatter(
    all_x, all_y,
    c=all_z,
    cmap='viridis',     # pode testar 'plasma' ou 'coolwarm'
    s=70,
    alpha=0.9,
    edgecolor='k',
    linewidth=0.4
)

# adiciona a barra de cor (escala angular)
cbar = plt.colorbar(sc)
cbar.set_label('Crank Angle [°]', fontsize=11)

# limites e linha 1:1
if all_x.size:
    lo = float(min(all_x.min(), all_y.min()))
    hi = float(max(all_x.max(), all_y.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    x_id = np.linspace(lo - pad, hi + pad, 50)
    plt.plot(x_id, x_id, 'k--', lw=1.2, label='Identity (y=x)')

# linha de regressão global
if can_fit:
    plt.plot(all_x, slope * all_x + intercept, 'r-', lw=1.5, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

# texto de métricas no canto
plt.text(0.05, 0.93,
         f"$R^2$ = {r_squared:.3f}\nRMSE = {rmse:.3f}",
         transform=plt.gca().transAxes,
         fontsize=10,
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

# rótulos e layout
plt.xlabel('Predicted Torque-time Integrals [N·m·s]', fontsize=12)
plt.ylabel('Measured Torque-time Integrals [N·m·s]', fontsize=12)
plt.title('Measured vs Predicted Torque-time Integrals\nColored by Crank Angle', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('Param_iden_cycling_Brazil/opt_sol_results/images/TTI_results.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
