import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

# ==== config ====
JSON_ARQ = "pickles/torques_crankset.json"   # caminho do JSON salvo
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
    regression_line = slope * np.array(integrals_pdct_list) + intercept
else:
    slope = intercept = r_squared = rmse = np.nan
    regression_line = None

# ==== plot ====
plt.figure(figsize=(10, 6))

unique_angles = sorted(set(angles_list))
unique_subjects = sorted(set(subjects_list))

# cores por ângulo / marcadores por músculo
colors = plt.cm.tab20b(np.linspace(0, 1, len(unique_angles)))
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
angle_color_map = {angle: color for angle, color in zip(unique_angles, colors)}
subject_marker_map = {subject: markers[i % len(markers)] for i, subject in enumerate(unique_subjects)}

# scatter (um ponto por par independente)
for msrd, pdct, ang, subj in zip(integrals_msrd_list, integrals_pdct_list, angles_list, subjects_list):
    plt.scatter(pdct, msrd,
                color=angle_color_map[ang],
                marker=subject_marker_map[subj])

# limites e linha 1:1
all_x = np.asarray(integrals_pdct_list, dtype=float)
all_y = np.asarray(integrals_msrd_list, dtype=float)
if all_x.size:
    lo = float(min(all_x.min(), all_y.min()))
    hi = float(max(all_x.max(), all_y.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    x_id = np.linspace(lo - pad, hi + pad, 50)
    plt.plot(x_id, x_id)  # identidade (y=x)

# reta de regressão global (se possível)
if can_fit:
    plt.plot(integrals_pdct_list, regression_line, color='green',
             label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

# legendas separadas (ângulo e músculo)
angle_legend_entries = [plt.Line2D([0], [0], color=angle_color_map[a], lw=4) for a in unique_angles]
subject_legend_entries = [plt.Line2D([0], [0], color='black',
                                     marker=subject_marker_map[s], linestyle='None') for s in unique_subjects]
combined_legend_entries = angle_legend_entries + subject_legend_entries
combined_legend_labels = [f'{a}°' for a in unique_angles] + [s for s in unique_subjects]
plt.legend(combined_legend_entries, combined_legend_labels,
           loc='lower right', ncol=1, fontsize='small', frameon=False)

# estética + textos
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('Predicted Torque-time Integrals [N·m·s]')
plt.ylabel('Measured Torque-time Integrals [N·m·s]')

txt_slope = f'{slope:.2f}' if np.isfinite(slope) else 'n/a'
txt_r2    = f'{r_squared:.2f}' if np.isfinite(r_squared) else 'n/a'
txt_rmse  = f'{rmse:.2f}' if np.isfinite(rmse) else 'n/a'
plt.text(0.05, 0.95, f'Slope: {txt_slope}', transform=ax.transAxes, va='top', fontsize='large')
plt.text(0.05, 0.91, f'$R^2$: {txt_r2}',   transform=ax.transAxes, va='top', fontsize='large')
plt.text(0.05, 0.86, f'RMSE: {txt_rmse}',  transform=ax.transAxes, va='top', fontsize='large')

plt.tight_layout()
# plt.savefig('TTI_results.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()