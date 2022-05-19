# NOTE: See Analysis, Modeling, and Simulation of the Accuracy of Continuous Glucose Sensors
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

total_time = 60 * 24  # min in a day
MDL_sample_time = 15.0  # min
sample_time = 5.0  # min
xi = -5.471
delta = 1.6898
gamma = -0.5444
lam = 15.96
PACF = 0.7

noise_T = 1 + int(total_time / MDL_sample_time)
noise_t = np.arange(0, noise_T) * MDL_sample_time
e = np.random.randn(noise_T)
for i in range(1, noise_T):
    e[i] = (e[i - 1] + e[i]) * PACF
eps = xi + lam * np.sinh((e - gamma) / delta)

T = 1 + int(total_time / sample_time)
t = np.arange(0, T) * sample_time
interp_f = interp1d(noise_t, eps, kind="cubic")
noise = interp_f(t)

plt.ion()
fig, axes = plt.subplots(figsize=(8, 6), nrows=2)
ax = axes[0]
ax.plot(noise_t, eps, "-o", label="sampled eps", alpha=0.2, markersize=3)
ax.plot(t, noise, "-o", label="cgm noise", alpha=0.2, markersize=3)

ax = axes[1]
ax.plot(t, 150 + noise + 75 * np.sin(np.arange(T) / (2 * 3.14 * 2)), label="cgm noisy")
ax.plot(t, 150 + +75 * np.sin(np.arange(T) / (2 * 3.14 * 2)), label="cgm clean")
ax.set(ylim=(0, 400))
fig.legend()
input("")
