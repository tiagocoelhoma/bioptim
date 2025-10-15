import os
import pickle
import biorbd as brbd
import numpy as np
import matplotlib.pyplot as plt

random_parameters = [[-2044517.3628, -15361055.1532, -45683433.1537, -67236448.1273, -48992455.0222, -14135456.3528],
                     0.1000000099977404, 0.20000000999764617, 0.01999999000782689]


def get_coefficients_sqrd(config, a_scale_values, angles, print_flag=True):
    # Convert knee angles from deg to rad
    # angles_deg = [float(angle[2]) for angle in angles]  # e.g. [[.., .., 145], ...]
    # x_data = np.radians(angles)
    x_data = angles

    # Degree of polynomial to fit
    if config[:-1] == 'bflh140_':
        degree = 4
    else:
        degree = 4

    # Fit polynomial to sqrt of data
    sqrt_y = np.sqrt(a_scale_values)
    coeffs = np.polyfit(x_data, sqrt_y, degree)

    def positive_poly(x):
        # Clip the polynomial result to avoid squaring negatives
        poly_val = np.polyval(coeffs, x)
        return np.clip(poly_val, 0, None) ** 2

    if print_flag:
        x_fit = np.linspace(min(x_data), max(x_data), 1000)
        y_fit = positive_poly(x_fit)

        plt.figure(figsize=(8, 5))
        plt.plot(x_fit, y_fit, label="(Poly(x))² fit", linewidth=2)
        plt.scatter(x_data, a_scale_values, color='red', label="Original data")
        plt.title("Positive Polynomial Fit in Radians")
        plt.xlabel("Knee angle [rad]")
        plt.ylabel("a_scale")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("Squared polynomial coefficients (use in np.polyval):")
    print(coeffs)

    coefficients_text = f"Coefficients {config}: ["
    for i in range(degree + 1):
        coefficients_text += f"{coeffs[i]:.4f}, "
    coefficients_text = coefficients_text[:-2] + "]"
    print(coefficients_text)

    coeff_formatted_list = [round(elem, 5) for elem in coeffs.tolist()]
    return coeff_formatted_list


def get_coefficients(config, a_scale_values, angles, print_flag=True):

    angles_deg = np.array(angles)
    x_data = (180 - angles_deg) * np.pi / 180
    # degree = len(angles_deg) - 1
    degree = 3
    coeffs = np.polyfit(x_data, a_scale_values, degree)

    if print_flag:
        x_fit = np.linspace(min(x_data), max(x_data), 1000)
        y_fit = np.poly1d(x_fit)

        plt.figure(figsize=(8, 5))
        plt.plot(x_fit, y_fit, label="Poly(x) fit", linewidth=2)
        plt.scatter(x_data, a_scale_values, color='red', label="Original data")
        plt.title("Polynomial Fit")
        plt.xlabel("x = (180 - angle) * π / 180 [rad]")
        plt.ylabel("a_scale")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("Polynomial coefficients (use in np.polyval):")
    print(coeffs)

    coefficients_text = f"Coefficients {config}: ["
    for i in range(degree + 1):
        coefficients_text += f"{coeffs[i]:.4f}, "
    coefficients_text = coefficients_text[:-2]  # Remove the last '+'
    coefficients_text += "]"
    print(coefficients_text)

    coeff_formatted_list = [round(elem, 5) for elem in coeffs.tolist()]
    return coeff_formatted_list

def load_pickle(filename):
    if not os.path.exists(filename):
        print(f"Arquivo não encontrado: {filename}, criando novo.")
        return {}
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, 'wb') as f:
            pickle.dump({}, f)
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def update_parameters_value(filename, data_entry, new_value):

    # just adding a random value if necessary (when muscles are not implemented yet)
    new_value = random_parameters if new_value is None else new_value

    data = load_pickle(filename)
    data[data_entry] = new_value
    save_pickle(filename, data)
    print(f"Dado salvo com chave '{data_entry}' em: {filename}")
    '''
    import os
    os.path.exists('../../opt_sol_results/pickles/prediction.pkl')
    '''


def init_model_parameters(model_path, filename):

    num_model = brbd.Model(model_path)
    mus_names = num_model.muscleNames()
    muscle_params = read_params(model_path, filename)
    for idx, mn in enumerate(mus_names):
        m = mn.to_string()
        if m not in muscle_params.keys():
            update_parameters_value(filename, m, None)
            muscle_params = read_params(model_path, filename)

def copy_params_right_to_left(model_path, filename, option=1):
    num_model = brbd.Model(model_path)
    mus_names = num_model.muscleNames()
    data = load_pickle(filename)
    for idx, mn in enumerate(mus_names):
        m = mn.to_string()
        # copy parameters from the right leg, same muscle
        if option==1:
            update_parameters_value(filename, f'{m[:-2]}_l', data[f'{m[:-2]}_r'])
        elif option==2:
            update_parameters_value(filename, f'{m[:-2]}_r', data[f'{m[:-2]}_l'])


def read_params(model_path, filename):
    num_model = brbd.Model(model_path)
    mus_names = num_model.muscleNames()
    try:
        data = load_pickle(filename)
    except FileNotFoundError:
        print(f"No data file {filename} yet, initializing parameters with random data...")
        data = {}
        for idx, mn in enumerate(mus_names):
            data.update({mn.to_string(): random_parameters})
        save_pickle(filename, data)
        data = load_pickle(filename)

    # check if all muscles have parameters
    for idx, mn in enumerate(mus_names):
        m = mn.to_string()
        if m not in data.keys():
            # copy parameters from the other leg, same muscle
            for i, j in enumerate(data.keys()):
                if j[:-1] == m[:-1]:
                    update_parameters_value(filename, m, data[j])
                    break
        # if there is no muscle on the other leg, fill with random params
        # (pay attention, because it should have had it defined)
        if m not in data.keys():
            update_parameters_value(filename, m, None)

    print("Read Params:", data)
    return data


def main(model_path, filename, parameters):
    # for evaluation only
    num_model = brbd.Model(model_path)
    mus_names = num_model.muscleNames()
    a_params = [5879.026, 11379.837, 11649.113, 11459.837, 9520.713, 3946.813]
    angles = [120, 110, 100, 90, 80, 70]
    muscle_params = read_params(model_path, filename)
    for idx, mn in enumerate(mus_names):
        m = mn.to_string()
        muscle_params[m][0] = get_coefficients(m, a_params, angles)
        update_parameters_value(filename, m, muscle_params[m])
    updated_data = read_params(model_path, filename)
    # print("Updated A param:\n", updated_data)

    tau_1_param, tau_2_param, km_param = 0.01, 0.02, 0.8
    muscle_params = read_params(model_path, filename)
    for idx, mn in enumerate(mus_names):
        m = mn.to_string()
        muscle_params[m][1] = tau_1_param
        muscle_params[m][2] = tau_2_param
        muscle_params[m][3] = km_param
        update_parameters_value(filename, m, muscle_params[m])
    updated_data = read_params(model_path, filename)
    # print("Updated other params:\n", updated_data)

if __name__ == '__main__':
    # model_path = f"../models/legcyc_rectfem_r.bioMod"
    model_path = f"../../../models/legcyc_hams_r.bioMod"
    # # just adding some random values for creating a test file and evaluate the model parameter handler
    # param_test = [[-1.92, -8.28, -11.900, -4.0448, 4.61428], 0.12136, 0.13586, 0.8]
    # main(model_path, 'pickles/test_file.pkl', param_test)
    # read_params(model_path, 'pickles/legcyc_rectfem_r_params_.pkl')
    # # read_params(model_path, 'pickles/legcyc_rectfem_r_params.pkl')
    # read_params(model_path, filename =f'../../opt_sol_results/pickles/cycling_muscles_params.pkl')
    # update_parameters_value(filename =f'../../opt_sol_results/pickles/cycling_muscles_params.pkl',
    #                         data_entry='bflh140_r',
    #                         new_value = [[253.7, 89.297, -1170.132, -1517.832, 158.36], 0.09996741388775505, 0.13401139582560484, 0.02001530042206482])
    # read_params(model_path, filename=f'../../opt_sol_results/pickles/cycling_muscles_params.pkl')