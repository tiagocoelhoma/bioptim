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

def main(biorbd_model_path, muscle_params, knee_range=[0.8, 2.3]):
    # Plot
    model = BiorbdModel(biorbd_model_path)
    filtered_params = {m: muscle_params[m] for m in model.muscle_names if m in muscle_params}
    # filtered_params.keys()
    ncols=int(len(filtered_params)/2)
    # Define the x range
    x = np.linspace(knee_range[0], knee_range[1], 500)
    fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(10, 6))

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


    for m, (muscle, coeffs) in enumerate(filtered_params.items()):
        y = np.array([poly_squared(xi, coeffs[0]) for xi in x])
        idx_0, idx_1 = m//ncols, m%ncols
        axs[idx_0, idx_1].plot(x, y, label=muscle, color=colors[muscle])
        axs[idx_0, idx_1].set_title(f'{muscle}')
        axs[idx_0, idx_1].axhline(
            y=15000, color="gray", linestyle="--", linewidth=1
        )
        axs[idx_0, idx_1].axhline(
            y=0, color="gray", linestyle="--", linewidth=1
        )
        axs[idx_0, idx_1].axhline(
            y=50000, color="red", linestyle=":", linewidth=1
        )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle("Squared Polynomial Curves for Muscles (y = (poly(x))²)", fontsize=14)

    plt.show()

if __name__ == "__main__":
    main()







