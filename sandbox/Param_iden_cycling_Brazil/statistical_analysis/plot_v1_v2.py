import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 90, 500)
for V1, V2 in [(0.001, 0.01), (0.002, 0.03), (0.003, 0.05)]:
    G_theta = V1 * theta * np.exp(-V2 * theta)
    plt.plot(theta, G_theta, label=f'V1={V1}, V2={V2}')
plt.xlabel('θ (deg)')
plt.ylabel('G(θ)')
plt.legend()
plt.title('Exploração da função G(θ) = V1·θ·exp(-V2·θ)')
plt.grid()
plt.show()
