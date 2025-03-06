import numpy as np
import matplotlib.pyplot as plt

# Generate a simple sine wave
t = np.linspace(0, 1, 1000)  # 1 second, 1000 samples
freq = 5  # 5 Hz
signal = np.sin(2 * np.pi * freq * t)

# Plot it
plt.plot(t, signal)
plt.title("5 Hz Sine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("sine_wave.png")  # Save to file
plt.close()