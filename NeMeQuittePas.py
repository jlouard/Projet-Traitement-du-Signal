import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as sig

B = 0.5
f_c = 50
N = 1000
dt = 0.1

T = np.linspace(-6, 6, N)
freqs = np.arange(1000)/1000*B

x = np.cos(2*np.pi*B*T) + 2*np.cos(2*np.pi*2.4*B*T + 3)

x_m = np.cos(2*np.pi*f_c*T)*x

x_md = np.cos(2*np.pi*f_c*T)*x_m

def FiltrePB(y, dt, N_filter, fc):
    num_low, den_low = sig.butter(N_filter, fc, btype='lowpass', fs=1/dt)
    y_low = sig.filtfilt(num_low, den_low, y)
    return y_low

x_mdf = FiltrePB(x_md, 0.1, 2, 2*B)

# Affichage
plt.figure(figsize = (15, 5))

plt.subplot(1, 2, 1)
plt.plot(x)
plt.plot(x_m)
plt.plot(x_md)
plt.plot(x_mdf)

plt.subplot(1, 2, 2)
plt.plot(freqs[:N//2], np.abs(np.fft.fft(x)[:N//2]))
plt.plot(freqs[:N//2], np.abs(np.fft.fft(x_m)[:N//2]))
plt.plot(freqs[:N//2], np.abs(np.fft.fft(x_md)[:N//2]))
plt.plot(freqs[:N//2], np.abs(np.fft.fft(x_mdf)[:N//2]))
plt.title('spectre')

plt.show()