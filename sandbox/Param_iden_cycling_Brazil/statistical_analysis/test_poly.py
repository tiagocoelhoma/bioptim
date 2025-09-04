# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the x range
# x = np.linspace(0.8, 2.1, 500)
#
# # Define the polynomial coefficients
# coeffs_3rd = [-920.26837, 4067.34271, -5807.54397, 2747.26059]
# coeffs_4th = [-2857.38602, 16182.08388, -33604.45233, 30345.73234, -9999.43325]
#
# # Evaluate the polynomials
# y3 = np.polyval(coeffs_3rd, x)
# y4 = np.polyval(coeffs_4th, x)
#
# # Plot
# plt.figure(figsize=(10, 6))
# plt.plot(x, y3, label='3rd order polynomial', color='blue')
# plt.plot(x, y4, label='4th order polynomial', color='red', linestyle='--')
#
# # Add labels and legend
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Polynomial Curves')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Define the x range
x = np.linspace(0.8, 2.1, 500)

# Define coefficients
muscle_polys = {
    # 'rect_fem_r': [-2857.38602, 16182.08388, -33604.45233, 30345.73234, -9999.43325],
    'rect_fem_r': [-3341.93786, 21354.2006, -50284.67265, 51844.00188, -19773.58173],
    # 'vasmed_r': [2077.58265, -11845.71106, 24967.97162, -23027.57252, 7902.8386],
    # 'bflh140_r': [2458.40138, -13921.07477, 29238.88064, -26993.60358, 9265.28969],
    # 'bflh140_l': [-4547.5274, 23534.5909, -45191.6344, 38138.3809, -11902.4418],
}

# Define function to compute squared polynomial
def poly_squared(x, coeffs):
    poly_val = sum(c * x**(len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
    return poly_val**2

# Plot
plt.figure(figsize=(10, 6))

colors = {'rect_fem_r': 'red', 'vasmed_r': 'green', 'bflh140_r': 'blue', 'bflh140_l': 'black'}

for muscle, coeffs in muscle_polys.items():
    y = np.array([poly_squared(xi, coeffs) for xi in x])
    plt.plot(x, y, label=muscle, color=colors[muscle])

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y = (poly(x))²')
plt.title('Squared Polynomial Curves for Muscles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
