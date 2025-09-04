import numpy as np
import matplotlib.pyplot as plt

# === Your data ===
# angles_deg = np.array([70, 80, 90, 100, 110, 120])
angles_deg = np.array([60, 75, 90, 105, 120, 155])
# a_scale_values = np.array([
#     7387.902043965209,
#     1409.1857974363413,
#     953.4974326531005,
#     987.8884427809242,
#     754.8649695914827,
#     364.6133808065618,
# ])
a_scale_values = np.array([6225.863488061236, 1899.3175693360274, 1320.9639302035748, 1415.2212806877492, 1111.1449496436294, 648.5715801272311])

# === Preprocess to match your Bioptim model ===
# (180 - angle) * pi / 180 → angle in radians, flipped
x_data = (180 - angles_deg) * np.pi / 180

# === Fit the square root of your data ===
degree = len(angles_deg) - 2  # degree 6
sqrt_y = np.sqrt(a_scale_values)
coeffs = np.polyfit(x_data, sqrt_y, degree)
# coeffs = [-514.89349, 4337.46459, -13909.84529, 21427.14638, -15924.00865, 4611.03791]  # RF
# coeffs = [0, 0, -218.06225, 727.42658, -804.45493, 337.79308]                           # HAM
# # coeffs = [-542.50233, 4508.0429, -14329.14381, 21943.98066, -16247.66918, 4692.93653] # Coefficients rect_fem_r
# # coeffs = [0, 0, -421.16044, 1347.67588, -1423.98292, 604.16395]                           # HAM
# coeffs = [-542.50233, 4508.0429, -14329.14381, 21943.98066, -16247.66918, 4692.93653]
coeffs = [-9140.66636, 65256.67817, -184489.01587, 258087.88695, -178561.02191, 48913.40928] # RF
coeffs = [721.23884, -2688.36002, 3352.98832, -1275.25342]
coeffs = [-171.71679534,  482.91403443, -415.53465775,  161.97051981]
coeffs = [-2669.4529, 18830.0767, -52270.6487, 71502.9956, -48294.0308, 12983.1377]
coeffs = [-483.1449, 2275.0135, -3470.5428, 1916.8521]

coeffs = [-2669.45289, 18830.07673, -52270.64874, 71502.99558, -48294.03077, 12983.1377] # rect_fem_r

# coeffs = [-2114.87661, 8622.20453, -11516.65504, 5174.72476] # vasmed_r
# coeffs = [36.5982, -212.43201, 264.94784, 25.92596] #bflh140_r

# coeffs = [-2200.76324, 15836.03392, -44955.71673, 63023.33497, -43684.94333, 12044.84212] # rect_fem_r + vastuaslat
coeffs = [-1995.05129, 14344.38493, -40686.69493, 56988.87941, -39467.19126, 10872.89719] # rect_fem_r + vastuaslat
coeffs = [-1995.05129, 14344.38493, -40686.69493, 56988.87941, -39467.19126, 10872.89719]

# [-18.4326, 215.54448, -480.88719, 366.20603]
coeffs = [-1360.0253, 9567.1157, -26505.0904, 36235.2379, -24504.1579, 6613.0525]# rect_fem_r
coeffs = [-46.5723, 278.8797, -492.4886, 323.6371]  # vasmed_r
coeffs = [-1360.02531, 9567.11571, -26505.09044, 36235.23789, -24504.15795, 6613.05246]
coeffs = [-4900.31189, 33094.70958, -88658.71006, 117806.29578, -77680.19418, 20383.55853]
coeffs = [-85.6783, 467.3867, -784.4138, 485.5037]
coeffs = [-1815.80135, 12790.62943, -35443.13248, 48382.2769, -32597.52521, 8737.74675]
x_data = (180 - angles_deg) * np.pi / 180
# === Define the positive polynomial ===
def positive_poly(x):
    return (np.polyval(coeffs, x)) ** 2

# === Plotting ===
# angles_deg_test = np.array([70, 80, 90, 100, 110, 120])
# x_data = (180 - angles_deg_test) * np.pi / 180
# x_fit = np.linspace(min(x_data), max(x_data), 1000)
x_fit = np.linspace(0.70, 2.10, 1000) # range of motion
y_fit = positive_poly(x_fit)

plt.figure(figsize=(8, 5))
plt.plot(x_fit, y_fit, label="(Poly(x))² fit", linewidth=2)
plt.scatter(x_data, a_scale_values, color='red', label="Original data")
plt.title("Positive Polynomial Fit")
plt.xlabel("x = (180 - angle) * π / 180 [rad]")
plt.ylabel("a_scale")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Print coefficients for use in Bioptim ===
print("Squared polynomial coefficients (use in np.polyval):")
print(coeffs)
