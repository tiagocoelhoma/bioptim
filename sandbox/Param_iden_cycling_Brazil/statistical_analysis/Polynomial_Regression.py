import numpy as np
import matplotlib.pyplot as plt


from Param_iden_cycling.ocp_param_indent.common.params_handler import read_params
from Param_identification.data_exp.data_evaluation import legs, a_scaling_params

# to be defined by the user
angles = np.array([70, 80, 90, 100, 110, 120])  # should be the angles used in the parameter identification
# angles = np.array([120, 110, 100, 90, 80, 70])  # should be the angles used in the parameter identification
# angles = np.array([60, 75, 90, 105, 120, 135])
# angles = np.array([75, 90, 105, 120])
a_scale_values = np.array([0, 30267.443700561653, 16322.67822188569, 13879.914498463666, 20860.76781949795, 0])
a_scale_values = np.array([1956.3687848119223, 2445.2043026048536, 1959.049150073414, 2428.874939868554, 2761.7519490977797, 16931.725599728863])
config = 'teste'


x_angles = np.linspace(min(angles), max(angles), 1000)
x = []
for i in range(len(angles)):
    x.append((180 - angles[i]) * np.pi / 180)       # 70 > 120 (converted to rad) according to the model fes-cycling

fig, ax1 = plt.subplots(1, 2, figsize=(10, 4))

degree = len(x) - 1
coefficients = np.polyfit(x, a_scale_values, degree)        # Fit polynomial curve to the data

# coeffs = [-2669.45289, 18830.07673, -52270.64874, 71502.99558, -48294.03077, 12983.1377] # rect_fem_r
# coeffs = [-2114.87661, 8622.20453, -11516.65504, 5174.72476] # vasmed_r
# coeffs = [36.5982, -212.43201, 264.94784, 25.92596] #bflh140_r

# coeffs = [-2200.76324, 15836.03392, -44955.71673, 63023.33497, -43684.94333, 12044.84212] # rect_fem_r + vastuaslat


poly_function = np.poly1d(coefficients)                     # Create a polynomial function using the coefficients
xx = np.linspace(min(x), max(x), 1000)                  # Generate x values for plotting the curve
xx = np.linspace(0.7, 2.1, 1000)                  # Generate x values for plotting the curve
yy = poly_function(xx)                                        # Evaluate the polynomial function at these x values

ax1[0].plot(angles, a_scale_values, 'ro')
ax1[0].plot(x_angles, yy, '-.', label=f'{config}')

# plt.title('Fitted Curve with Polynomial Regression', fontsize=16)
ax1[0].set_xlabel('knee angles')
ax1[0].set_ylabel('A scale')
ax1[0].patch.set_facecolor('lavender')
ax1[0].grid(True)
ax1[0].legend(loc='upper right', fontsize='x-small')

yy_normalized = (yy - yy.min()) / (yy.max() - yy.min())
y_normalized = (a_scale_values - np.min(a_scale_values)) / (np.max(a_scale_values) - np.min(a_scale_values))
ax1[1].plot(angles, y_normalized, 'ro')
ax1[1].plot(x_angles, yy_normalized, '-.', label=f'test')
ax1[1].set_xlabel('knee angles')
ax1[1].set_ylabel('Normalized A scale')
ax1[1].tick_params(axis='y', labelcolor='blue')
ax1[1].patch.set_facecolor('lavender')
ax1[1].grid(True)
ax1[1].legend(loc='upper right', fontsize='x-small')

coefficients_text = f"Coefficients: ["
for i in range(degree + 1):
    coefficients_text += f"{coefficients[i]:.3f}, "
coefficients_text = coefficients_text[:-2]  # Remove the last '+'
coefficients_text += "]"
print(coefficients_text)

plt.show()
