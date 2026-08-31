# Sim.-Reservat-rio

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, expi

# parametros
k = 1e-13       # permeabilidade (m²)
phi = 0.2       # porosidade
mu = 1e-3       # viscosidade (Pa.s)
ct = 1e-9       # compressibilidade (1/Pa)
p0 = 200e5      # pressão inicial (Pa)
pw = 150e5      # pressão no poço (Pa)
qw = 0.001      # vazão (m³/s)
A = 10000       # área (m²)
L = 1000        # comprimento (m)
h = 20          # espessura (m)
rw = 0.1        # raio do poço (m)
re = 1000       # raio externo (m)

def analitica1(n, L, eta, t, x):
    return (np.exp(-((n * np.pi / L)**2) * eta * t) / n) * np.sin(n * np.pi * x / L)

def analitica2(x, t, eta, L, p0, qw, mu, k, A):
    parte1 = (np.sqrt(4 * eta * t / (np.pi * L**2))* np.exp(-x**2 / (4 * eta * t)))
    parte2 = ((x / L)* erfc(x / np.sqrt(4 * eta * t)))
    return p0 - (qw * mu * L / (k * A)) * (parte1 - parte2)

def analitica3(r, t, phi, mu, ct, k, h, p0, qw):
    arg = (phi * mu * ct * r**2) / (4 * k * t)
    return p0 - (qw * mu / (4 * np.pi * k * h)) * (-expi(-arg))

def analitica4(r, t, phi, mu, ct, k, h, p0, qw, re, rw):
    t1 = (2 * k * t) / (phi * mu * ct * re**2)
    t2 = (-np.log(r / rw)+ 0.5 * (r / re)**2+ np.log(re / rw)- 0.75)
    return p0 - (qw * mu / (2 * np.pi * k * h)) * (t1 + t2)

eta = k / (phi * mu * ct)

x = np.linspace(0.1, L, 200)
r = np.linspace(rw, re, 200)

tempos = [3600, 86400, 864000] # 1h, 1 dia, 10 dias

#graficodo linear finito
plt.figure(1, figsize=(8, 5))
for t in tempos:
    soma = np.zeros(len(x))
    for n in range(1, 150):
        soma += analitica1(n, L, eta, t, x)
    p = (p0 - pw) * (x / L + (2 / np.pi) * soma) + pw
    plt.plot(x, p/1e5, label=f't = {round(t/86400, 2)} dias')

plt.title('Linear Finito (P-P)')
plt.xlabel('Distância x (m)')
plt.ylabel('Pressão (bar)')
plt.grid()
plt.legend()


#grafico linear semi
plt.figure(2, figsize=(8, 5))
for t in tempos:
    p = analitica2(x, t, eta, L, p0, qw, mu, k, A)
    plt.plot(x, p/1e5, label=f't = {round(t/86400, 2)} dias')

plt.title('Linear Semi-Infinito (Q-P)')
plt.xlabel('Distância x (m)')
plt.ylabel('Pressão (bar)')
plt.grid()
plt.legend()


#grafico r. transiente
plt.figure(3, figsize=(8, 5))
for t in tempos:
    p = analitica3(r, t, phi, mu, ct, k, h, p0, qw)
    plt.plot(r, p/1e5, label=f't = {round(t/86400, 2)} dias')

plt.title('Radial Transiente')
plt.xlabel('Raio r (m)')
plt.ylabel('Pressão (bar)')
plt.xscale('log')
plt.grid()
plt.legend()


# grafico r.pseudopermanente
plt.figure(4, figsize=(8, 5))
t_pseudo = [31536000, 31536000*2, 31536000*3] # 1, 2, 3 anos
for t in t_pseudo:
    p = analitica4(r, t, phi, mu, ct, k, h, p0, qw, re, rw)
    plt.plot(r, p/1e5, label=f't = {int(t/31536000)} anos')

plt.title('Radial Pseudopermanente')
plt.xlabel('Raio r (m)')
plt.ylabel('Pressão (bar)')
plt.xscale('log')
plt.grid()
plt.legend()

plt.show()
