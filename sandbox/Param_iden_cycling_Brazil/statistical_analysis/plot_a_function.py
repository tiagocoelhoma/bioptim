import numpy as np
import matplotlib.pyplot as plt

from bioptim import BiorbdModel

# Define coefficients
muscle_polys = {
    'rect_fem_r': [[0.0, -3615.85723, 26997.47466, -74512.17288, 90225.05965, -40357.58898],
                   0.009999990000012463, 0.192312298691983, 0.44033137269022055],
     'vasmed_r': [[0.0, -2186.41896, 15228.41864, -39311.39503, 44684.42858, -18797.18749],
                  0.009999990000021658, 0.20000001000171602, 0.5556868349146159],
     'rect_fem_l': [[0.0, 4148.94356, -29238.5829, 77048.64526, -89837.57164, 39160.61796],
                    0.009999990000014582, 0.2000000099999483, 0.4302961523991153],
     'vasmed_l': [[0.0, 3364.59387, -22690.88231, 57181.05191, -63744.66224, 26613.25594],
                  0.009999990000013411, 0.20000000999994902, 0.45190654258293717],
     'bflh140_r': [[0.0, 278227.93845, -1493263.36132, 2982789.04788, -2628023.14732, 861781.5701],
                   0.009999990000096198, 0.13491476887093923, 0.8000000099907685],
     'bflh140_l': [[0.0, 22986.6983, -134076.96801, 291810.57142, -280937.97825, 101075.95653],
                   0.009999990000085059, 0.11303012918867036, 0.800000009990738],
     'vast_lat_r': [[-2044517.3628, -15361055.1532, -45683433.1537, -67236448.1273, -48992455.0222, -14135456.3528],
                    0.1000000099977404, 0.20000000999764617, 0.01999999000782689],
     'vast_lat_l': [[-2044517.3628, -15361055.1532, -45683433.1537, -67236448.1273, -48992455.0222, -14135456.3528],
                    0.1000000099977404, 0.20000000999764617, 0.01999999000782689],
     'bfsh140_r': [[-2044517.3628, -15361055.1532, -45683433.1537, -67236448.1273, -48992455.0222, -14135456.3528],
                   0.1000000099977404, 0.20000000999764617, 0.01999999000782689],
     'bfsh140_l': [[-2044517.3628, -15361055.1532, -45683433.1537, -67236448.1273, -48992455.0222, -14135456.3528],
                   0.1000000099977404, 0.20000000999764617, 0.01999999000782689],
}

# Define function to compute squared polynomial
def poly_squared(x, coeffs):
    poly_val = sum(c * x**(len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
    return poly_val**2
    # return np.polyval(coeffs, x)**2

def main(biorbd_model_path, muscle_params, knee_range=[0.5, 2.5]):
    # Plot
    model = BiorbdModel(biorbd_model_path)
    filtered_params = {m: muscle_params[m] for m in model.muscle_names if m in muscle_params}
    # filtered_params.keys()
    ncols=int(len(filtered_params)/2)
    # Define the x range
    x = np.linspace(knee_range[0], knee_range[1], 500)
    # fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(10, 6))

    colors = {'rect_fem_r': 'red',
              'rect_fem_l': 'orange',
              'vasmed_r': 'green',
              'vasmed_l': 'purple',
              'bflh140_r': 'blue',
              'bflh140_l': 'black',
               'vast_lat_r': 'black',
               'vast_lat_l': 'black',
               'bfsh140_r': 'black',
               'bfsh140_l': 'black',
              }


    # === FIGURA 1: curvas individuais ===
    fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(10, 6))
    axs = axs.flatten()

    all_y = {}  # armazenar curvas para depois normalizar

    for m, (muscle, coeffs) in enumerate(filtered_params.items()):
        y = np.array([poly_squared(xi, coeffs[0]) for xi in x])
        all_y[muscle] = y

        axs[m].plot(x, y, label=muscle, color=colors.get(muscle, 'black'))
        axs[m].set_title(muscle)
        axs[m].axhline(y=0, color="gray", linestyle="--", linewidth=1)
        axs[m].axhline(y=15000, color="gray", linestyle="--", linewidth=1)
        axs[m].axhline(y=50000, color="red", linestyle=":", linewidth=1)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle("Squared Polynomial Curves for Muscles (y = (poly(x))²)", fontsize=14)
    # plt.show()

    # === FIGURA 2: todas as curvas normalizadas ===
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for muscle, y in all_y.items():
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
        ax2.plot(x, y_norm, label=muscle, color=colors.get(muscle, 'black'))

    ax2.set_title("Normalized Squared Polynomial Curves (0–1 scale)")
    ax2.set_xlabel("Knee angle [rad]")
    ax2.set_ylabel("Normalized value")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig2.tight_layout()

    # plt.show()

    # === FIGURA 3: todas as curvas normalizadas (global max) ===
    # Calcula o maior valor entre todas as curvas
    global_max = max(np.max(y) for y in all_y.values())

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for muscle, y in all_y.items():
        y_norm = y / global_max
        ax2.plot(x, y_norm, label=muscle, color=colors.get(muscle, 'black'))

    ax2.set_title("Normalized Squared Polynomial Curves (global normalization)")
    ax2.set_xlabel("Knee angle [rad]")
    ax2.set_ylabel("Normalized Torque") #y / max(y_all)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.legend(
        loc="upper center",         # posição central no topo
        # bbox_to_anchor=(0.5, 1.15), # desloca para dentro/fora do gráfico
        ncol=6,                     # número de colunas (ajuste conforme nº de músculos)
        frameon=False,              # sem moldura (opcional)
        fontsize=8
    )

    fig2.tight_layout()
    # plt.show()

    # Adiciona área sombreada entre x = 1.3 e x = 1.9
    ax2.axvspan(1.3, 2.1, color='gray', alpha=0.2, label='Knee range 1.3–1.9 rad')
    plt.savefig('Param_iden_cycling_Brazil/opt_sol_results/images/normalized_curves.pdf', format='pdf', dpi=300,
                bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()







