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


""" D """

x_m = np.sin(2*np.pi*f_c*T)*x
x_md = np.sin(2*np.pi*f_c*T)*x_m
x_mdf = FiltrePB(x_md, 0.1, 2, B)

#Affichage du signal
plt.figure(figsize = (15, 7))
plt.subplot(1,2,1)
plt.plot(T, x, label='$x$')
plt.plot(T, x_m, label='$x_m$')
#plt.plot(T, x_md, label='$x_{md}$')
plt.plot(T, x_mdf, label='$x_{mdf}$')
plt.legend()

#Affichage du spectre
freqs = np.linspace(0,2*np.pi*(N-1)/N/dt,N)/2/np.pi
plt.subplot(1,2,2)
plt.plot(freqs[:N//5], np.abs(np.fft.fft(x)[:N//5]), label='Spectre de $x$')
plt.plot(freqs[:N//5], np.abs(np.fft.fft(x_m)[:N//5]), label='Spectre de $x_m$')
#plt.plot(freqs[:N//5], np.abs(np.fft.fft(x_md)[:N//5]), label='Spectre de $x_{md}$')
plt.plot(freqs[:N//5], np.abs(np.fft.fft(x_mdf)[:N//5]), label='Spectre de $x_{mdf}$')
plt.legend()




""" II """


x1 = np.cos(2*np.pi*B*T) + 2*np.cos(2*np.pi*1.4*B*T + 3)
xQ = np.sin(2*np.pi*1.2*B*T) + 1.5*np.cos(2*np.pi*0.8*B*T)

x_m = np.cos(2*np.pi*f_c*T)*x1 + np.sin(2*np.pi*f_c*T)*xQ

x1_md = np.cos(2*np.pi*f_c*T)*x_m
xQ_md = np.sin(2*np.pi*f_c*T)*x_m

x1_mdf = FiltrePB(x1_md, 0.1, 2, B)
xQ_mdf = FiltrePB(xQ_md, 0.1, 2, B)


#Affichage du signal
plt.figure(figsize = (15, 7))
plt.subplot(1,2,1)
plt.plot(T, x1, label='$x_1$')
plt.plot(T, xQ, label='$x_Q$')
#plt.plot(T, x_m, label='$x_m$')
#plt.plot(T, x1_md, label='$x1_{md}$')
#plt.plot(T, xQ_md, label='$xQ_{md}$')
plt.plot(T, x1_mdf, label='$x1_{mdf}$')
plt.plot(T, xQ_mdf, label='$xQ_{mdf}$')
plt.legend()

#Affichage du spectre
freqs = np.linspace(0,2*np.pi*(N-1)/N/dt,N)/2/np.pi
plt.subplot(1,2,2)
plt.plot(freqs[:N//2], np.abs(np.fft.fft(x1)[:N//2]), label='Spectre de $x_1$')
plt.plot(freqs[:N//2], np.abs(np.fft.fft(xQ)[:N//2]), label='Spectre de $x_Q$')
#plt.plot(freqs[:N//2], np.abs(np.fft.fft(x_m)[:N//2]), label='Spectre de $x_m$')
#plt.plot(freqs[:N//2], np.abs(np.fft.fft(x1_md)[:N//2]), label='Spectre de $x1_{md}$')
#plt.plot(freqs[:N//2], np.abs(np.fft.fft(xQ_md)[:N//2]), label='Spectre de $xQ_{md}$')
plt.plot(freqs[:N//2], np.abs(np.fft.fft(x1_mdf)[:N//2]), label='Spectre de $x1_{mdf}$')
plt.plot(freqs[:N//2], np.abs(np.fft.fft(xQ_mdf)[:N//2]), label='Spectre de $xQ_{mdf}$')
plt.legend()



""" III """

x = np.cos(2*np.pi*B*T) + 2*np.cos(2*np.pi*1.4*B*T + 3)
xchap = -sig.hilbert(x).imag

x_m = np.cos(2*np.pi*f_c*T)*x + np.sin(2*np.pi*f_c*T)*xchap

x_md = np.cos(2*np.pi*f_c*T)*x_m
xchap_md = np.sin(2*np.pi*f_c*T)*x_m

x_mdf = FiltrePB(x_md, 0.1, 2, B)
xchap_mdf = FiltrePB(xchap_md, 0.1, 2, B)


#Affichage du signal
plt.figure(figsize = (15, 7))
plt.subplot(1,2,1)
plt.plot(T, x, label='$x$')
plt.plot(T, xchap, label='$xchap$')
#plt.plot(T, x_m, label='$x_m$')
#plt.plot(T, x_md, label='$x_{md}$')
#plt.plot(T, xchap_md, label='$xchap_{md}$')
plt.plot(T, x_mdf, label='$x_{mdf}$')
plt.plot(T, xchap_mdf, label='$xchap_{mdf}$')
plt.legend()

#Affichage du spectre
freqs = np.linspace(0,2*np.pi*(N-1)/N/dt,N)/2/np.pi
plt.subplot(1,2,2)
plt.plot(freqs[:N//2], np.abs(np.fft.fft(x1)[:N//2]), label='Spectre de $x$')
plt.plot(freqs[:N//2], np.abs(np.fft.fft(xQ)[:N//2]), label='Spectre de $xchap$')
#plt.plot(freqs[:N//2], np.abs(np.fft.fft(x_m)[:N//2]), label='Spectre de $x_m$')
#plt.plot(freqs[:N//2], np.abs(np.fft.fft(x_md)[:N//2]), label='Spectre de $x1_{md}$')
#plt.plot(freqs[:N//2], np.abs(np.fft.fft(xchap_md)[:N//2]), label='Spectre de $xQ_{md}$')
plt.plot(freqs[:N//2], np.abs(np.fft.fft(x_mdf)[:N//2]), label='Spectre de $x_{mdf}$')
plt.plot(freqs[:N//2], np.abs(np.fft.fft(xchap_mdf)[:N//2]), label='Spectre de $xchap_{mdf}$')
plt.legend()


"""
SSB-SC :

Considérons en entrée un signal de la forme : $x(t) = Acos(2\pi B t)$

Avec la méthode de DSB-SC, on module le signal $x(t)$ en le multipliant par un cosinus de haute fréquence :

   \begin{array}{r c l}
   x_{m}(t)  & = & x(t)cos(2\pi f_c t) \\
   & = & Acos(2\pi B t)cos(2\pi f_c t) \\
   & = & \frac{A}{2} [cos(2\pi (f_c+B) t) + cos(2\pi (f_c-B) t)] \\
   \end{array}

Dans le cas du DSB-SC on avait donc 2 bandes à transmettre, $f_c-B$ et $f_c+B$. 
Pour économiser la bande passante, on cherche à sélectionner une seule bande du signal que l'on va transmettre dans le cas de la SSB-SC. Prenons par exemple la bande supérieure (Upper Band) de fréquence $f_c+B$.

On veut donc que :
   \begin{array}{r c l}
   x_{m}(t)  & = & Bcos(2\pi (f_c+B) t) \\
   & = & B[cos(2\pi f_c t)cos(2\pi B t) - sin(2\pi f_c t)sin(2\pi B t)] \\
   \end{array}

Si on définit $\hat x$ comme :
$\hat x(t) = Asin(2\pi B t)$, on peut alors reprendre le cas précédent avec $x_I = x$ et $x_Q = \hat x$

Par linéarité, on remarquera que le résultat se généralise pour toute somme dénombrable de signaux sinusoïdaux.

Pour définir $\hat x$, on utilisera la partie réelle de *sig.hilbert(x)*.
"""
