import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt



def get_coefficients_sqrd(a_scale_values, angles, print_flag=True):
    # Convert knee angles from deg to rad
    angles_deg = [float(angle) for angle in angles]  # e.g. [[.., .., 145], ...]
    x_data = np.radians(angles_deg)

    # Degree of polynomial to fit
    degree = 3

    # Fit polynomial to sqrt of data
    sqrt_y = np.sqrt(a_scale_values)
    coeffs = np.polyfit(x_data, sqrt_y, degree)

    def positive_poly(x):
        # Clip the polynomial result to avoid squaring negatives
        poly_val = np.polyval(coeffs, x)
        return np.clip(poly_val, 0, None) ** 2


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

    coefficients_text = f"Coefficients: ["
    for i in range(degree + 1):
        coefficients_text += f"{coeffs[i]:.4f}, "
    coefficients_text = coefficients_text[:-2] + "]"
    print(coefficients_text)

    coeff_formatted_list = [round(elem, 5) for elem in coeffs.tolist()]
    return coeff_formatted_list

q_init_list = [60, 70, 100, 120, 140]
a_scale_values = [4591.281935562991, 3044.7217235854314, 3861.4121837355665, 4293.008796358981, 3226.851684174334]

a_scale_values = [99.99999988925069, 627.857828329073, 976.5780747890808, 1316.337868102663, 1266.1137794239462, 1170.893981812462, 933.4650581291704, 855.1572358531979, 776.7716348409938, 440.7737100919783]


get_coefficients_sqrd(a_scale_values, q_init_list)