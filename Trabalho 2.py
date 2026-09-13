import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.linalg import solve_banded

# parametros (os mesmos da atv 1)
k = 1e-13       #permeabilidade (m2)
phi = 0.2       #porosidade
mu = 1e-3       #viscosidade (Pa.s)
ct = 1e-9       #compressibilidade (1/Pa)
p0 = 200e5      #pressao i
pw = 150e5      #pressao no poco
pe = 200e5      #pressao na fronteira externa
qw = 0.001      #vazao (m3/s)
A = 10000       #area
L = 1000        #comprimento

eta = k / (phi * mu * ct)

DIA = 86400.0

# funcoes analiticas da atv 1 para validacao
def analitica_pp(x, t, L, eta, pe, pw):
    soma = np.zeros(len(x))
    for n in range(1, 150):
        soma += (np.exp(-((n * np.pi / L)**2) * eta * t) / n) * np.sin(n * np.pi * x / L)
    return (pe - pw) * (x / L + (2 / np.pi) * soma) + pw

def analitica_qp(x, t, eta, L, p0, qw, mu, k, A):
    parte1 = np.sqrt(4 * eta * t / (np.pi * L**2)) * np.exp(-x**2 / (4 * eta * t))
    parte2 = (x / L) * erfc(x / np.sqrt(4 * eta * t))
    return p0 - (qw * mu * L / (k * A)) * (parte1 - parte2)

# resolve o sistema tridiagonal pelo scipy no formato de banda
def solver_banda(di, dp, ds, d):
    n = len(d)
    ab = np.zeros((3, n))
    ab[0, 1:] = ds
    ab[1, :] = dp
    ab[2, :-1] = di
    return solve_banded((1, 1), ab, d)

# erro maximo e rms em bar
def erros(p_num, p_ana):
    dif = (p_num - p_ana) / 1e5
    return np.abs(dif).max(), np.sqrt((dif**2).mean())



# discretizacao malha
N = 50
dx = L / N
x_num = np.linspace(dx/2, L - dx/2, N) # nós nos centroides
x_ana = np.linspace(0.1, L, 200)       # malhapro analitico

# discretizacao tempo
# estabilidade explicito Dirichlet: (1 - 4*eta*rx) >= 0 -> dt <= dx^2 / (4*eta)
dt = 100 # segundos
tempos_dias = [0.25, 1.0, 2.0]
rotulos = ['6 h', '1 d', '2 d']
passos_tempo = [int(d * DIA / dt) for d in tempos_dias]
max_passos = max(passos_tempo)

rx = dt / (dx**2)
eta_rx = eta * rx

print('eta =', round(eta, 4), 'm2/s')
print('dx =', dx, 'm | dt =', dt, 's | lambda =', round(eta_rx, 4))
print('dt maximo explicito =', round(dx**2 / (4*eta), 1), 's')
print('regime permanente P-P em', round(L**2 / (np.pi**2 * eta) / DIA, 2), 'dias')
print('semi-infinito valido ate', round(L**2 / (4*eta) / DIA, 2), 'dias')
for d, rot in zip(tempos_dias, rotulos):
    print('penetracao em', rot, '=', round(np.sqrt(4*eta*d*DIA)), 'm')

# Simulação Pressao-Pressao

p_exp_pp = np.ones(N) * p0
p_imp_pp = np.ones(N) * p0

# matriz tridiagonal implicito
diag_princ = np.ones(N) * (1 + 2 * eta_rx)
diag_princ[0] = 1 + 4 * eta_rx
diag_princ[-1] = 1 + 4 * eta_rx

diag_sup = np.ones(N-1) * (-eta_rx)
diag_sup[0] = -(4/3) * eta_rx

diag_inf = np.ones(N-1) * (-eta_rx)
diag_inf[-1] = -(4/3) * eta_rx

A_pp = diags([diag_inf, diag_princ, diag_sup], [-1, 0, 1], format='csc')
lu_pp = splu(A_pp)   # fatora uma vez so

res_exp_pp, res_imp_pp = {}, {}

for n in range(1, max_passos + 1):
    # explicito
    p_new = np.empty(N)
    p_new[0] = (1 - 4*eta_rx)*p_exp_pp[0] + (4/3)*eta_rx*p_exp_pp[1] + (8/3)*eta_rx*pw
    p_new[1:-1] = (1 - 2*eta_rx)*p_exp_pp[1:-1] + eta_rx*(p_exp_pp[2:] + p_exp_pp[:-2])
    p_new[-1] = (1 - 4*eta_rx)*p_exp_pp[-1] + (4/3)*eta_rx*p_exp_pp[-2] + (8/3)*eta_rx*pe
    p_exp_pp = p_new

    # implicito
    b = np.copy(p_imp_pp)
    b[0] += (8/3) * eta_rx * pw
    b[-1] += (8/3) * eta_rx * pe
    p_imp_pp = lu_pp.solve(b)

    if n in passos_tempo:
        dia = tempos_dias[passos_tempo.index(n)]
        res_exp_pp[dia] = np.copy(p_exp_pp)
        res_imp_pp[dia] = np.copy(p_imp_pp)

#simulação: vazão x pressão

p_exp_qp = np.ones(N) * p0
p_imp_qp = np.ones(N) * p0

#matriz tridiagonal implicito
diag_princ_q = np.ones(N) * (1 + 2 * eta_rx)
diag_princ_q[0] = 1 + eta_rx  # neumann na esquerda
diag_princ_q[-1] = 1 + 4 * eta_rx # dirichlet na direita

diag_sup_q = np.ones(N-1) * (-eta_rx)
diag_inf_q = np.ones(N-1) * (-eta_rx)
diag_inf_q[-1] = -(4/3) * eta_rx

A_qp = diags([diag_inf_q, diag_princ_q, diag_sup_q], [-1, 0, 1], format='csc')
lu_qp = splu(A_qp)

# termo fonte
termo_fonte_q = eta_rx * (qw * mu * dx) / (k * A)
res_exp_qp, res_imp_qp = {}, {}

for n in range(1, max_passos + 1):
    # explicito
    p_new = np.empty(N)
    p_new[0] = (1 - eta_rx)*p_exp_qp[0] + eta_rx*p_exp_qp[1] - termo_fonte_q
    p_new[1:-1] = (1 - 2*eta_rx)*p_exp_qp[1:-1] + eta_rx*(p_exp_qp[2:] + p_exp_qp[:-2])
    p_new[-1] = (1 - 4*eta_rx)*p_exp_qp[-1] + (4/3)*eta_rx*p_exp_qp[-2] + (8/3)*eta_rx*p0
    p_exp_qp = p_new

    #implicito
    b = np.copy(p_imp_qp)
    b[0] -= termo_fonte_q
    b[-1] += (8/3) * eta_rx * p0
    p_imp_qp = lu_qp.solve(b)

    if n in passos_tempo:
        dia = tempos_dias[passos_tempo.index(n)]
        res_exp_qp[dia] = np.copy(p_exp_qp)
        res_imp_qp[dia] = np.copy(p_imp_qp)

print('\nerro P-P (bar)')
for i, dia in enumerate(tempos_dias):
    p_ana = analitica_pp(x_num, dia * DIA, L, eta, pe, pw)
    e_exp = erros(res_exp_pp[dia], p_ana)
    e_imp = erros(res_imp_pp[dia], p_ana)
    print(rotulos[i], '- explicito: max', round(e_exp[0], 4), 'rms', round(e_exp[1], 4),
          '| implicito: max', round(e_imp[0], 4), 'rms', round(e_imp[1], 4))

print('\nerro Q-P (bar)')
for i, dia in enumerate(tempos_dias):
    p_ana = analitica_qp(x_num, dia * DIA, eta, L, p0, qw, mu, k, A)
    e_exp = erros(res_exp_qp[dia], p_ana)
    e_imp = erros(res_imp_qp[dia], p_ana)
    print(rotulos[i], '- explicito: max', round(e_exp[0], 4), 'rms', round(e_exp[1], 4),
          '| implicito: max', round(e_imp[0], 4), 'rms', round(e_imp[1], 4))

#comparacao explicito x implicito

print('\ncomparacao explicito x implicito (P-P, 1 dia)')
p_ana_1d = analitica_pp(x_num, 1 * DIA, L, eta, pe, pw)

for lam in [0.05, 0.15, 0.24, 1.0, 5.0, 25.0]:
    dt_c = lam * dx**2 / eta
    n_passos = int(np.ceil(1 * DIA / dt_c))
    dt_c = 1 * DIA / n_passos
    er = eta * dt_c / dx**2

    dpc = np.ones(N) * (1 + 2*er)
    dpc[0] = 1 + 4*er
    dpc[-1] = 1 + 4*er
    dsc = np.ones(N-1) * (-er)
    dsc[0] = -(4/3)*er
    dic = np.ones(N-1) * (-er)
    dic[-1] = -(4/3)*er
    lu_c = splu(diags([dic, dpc, dsc], [-1, 0, 1], format='csc'))

    # explicito so roda dentro do limite de estabilidade
    if lam <= 0.25:
        p = np.ones(N) * p0
        for n in range(n_passos):
            pn = np.empty(N)
            pn[0] = (1 - 4*er)*p[0] + (4/3)*er*p[1] + (8/3)*er*pw
            pn[1:-1] = (1 - 2*er)*p[1:-1] + er*(p[2:] + p[:-2])
            pn[-1] = (1 - 4*er)*p[-1] + (4/3)*er*p[-2] + (8/3)*er*pe
            p = pn
        erms = erros(p, p_ana_1d)[1]
        print('explicito: lambda', lam, '| dt', round(dt_c, 1), 's |', n_passos,
              'passos | rms', round(erms, 5), 'bar')

    # implicito
    p = np.ones(N) * p0
    for n in range(n_passos):
        b = np.copy(p)
        b[0] += (8/3) * er * pw
        b[-1] += (8/3) * er * pe
        p = lu_c.solve(b)
    erms = erros(p, p_ana_1d)[1]
    print('implicito: lambda', lam, '| dt', round(dt_c, 1), 's |', n_passos,
          'passos | rms', round(erms, 5), 'bar')

#teste do criterio de estabilidade

print('\nestabilidade do explicito (Pressão pressao, 1 dia)')
lambdas_teste = [0.10, 0.24, 0.45, 0.55]

for lam in lambdas_teste:
    dt_e = lam * dx**2 / eta
    n_passos = int(np.ceil(1 * DIA / dt_e))
    p = np.ones(N) * p0
    for n in range(n_passos):
        pn = np.empty(N)
        pn[0] = (1 - 4*lam)*p[0] + (4/3)*lam*p[1] + (8/3)*lam*pw
        pn[1:-1] = (1 - 2*lam)*p[1:-1] + lam*(p[2:] + p[:-2])
        pn[-1] = (1 - 4*lam)*p[-1] + (4/3)*lam*p[-2] + (8/3)*lam*pe
        p = pn
    #pressao dentro dos limites fisics
    ok = np.all(np.isfinite(p)) and p.min() > 0.9*pw and p.max() < 1.1*pe
    print('lambda', lam, '| dt', round(dt_e, 1), 's | p min',
          '{:.3e}'.format(p.min()/1e5), '| p max', '{:.3e}'.format(p.max()/1e5),
          '|', 'estavel' if ok else 'DIVERGIU')

#comparacao dos metodos de solucao do sistema linear

print('\nsolvers do sistema linear (implicito, pressão pressão, 1 dia)')
n_passos_s = int(1 * DIA / dt)

p_banda = np.ones(N) * p0
for n in range(n_passos_s):
    b = np.copy(p_banda)
    b[0] += (8/3) * eta_rx * pw
    b[-1] += (8/3) * eta_rx * pe
    p_banda = solver_banda(diag_inf, diag_princ, diag_sup, b)

p_lu = np.ones(N) * p0
for n in range(n_passos_s):
    b = np.copy(p_lu)
    b[0] += (8/3) * eta_rx * pw
    b[-1] += (8/3) * eta_rx * pe
    p_lu = lu_pp.solve(b)

print('diferenca maxima entre solve_banded e splu:',
      '{:.3e}'.format(np.abs(p_banda - p_lu).max()), 'Pa')

# graficos
marcadores = ['o', 's', '^']

#grafico 1: Pressão pressão
fig1, (ax1, ax1e) = plt.subplots(1, 2, figsize=(13, 5))
for idx, dia in enumerate(tempos_dias):
    t_seg = dia * DIA
    p_ana = analitica_pp(x_ana, t_seg, L, eta, pe, pw)
    ax1.plot(x_ana, p_ana/1e5, color='black', label='Analítica' if idx == 0 else "")
    ax1.scatter(x_num, res_exp_pp[dia]/1e5, marker=marcadores[idx], facecolors='none',
                edgecolors='r', s=40, label='Explícito' if idx == 0 else "")
    ax1.scatter(x_num, res_imp_pp[dia]/1e5, marker='x', color='b', s=40,
                label='Implícito' if idx == 0 else "")
    i_meio = np.argmin(np.abs(p_ana - (pw + pe)/2))
    ax1.annotate(rotulos[idx], (x_ana[i_meio], p_ana[i_meio]/1e5), fontsize=10,
                 textcoords='offset points', xytext=(8, -4))

    p_ana_n = analitica_pp(x_num, t_seg, L, eta, pe, pw)
    ax1e.plot(x_num, (res_exp_pp[dia] - p_ana_n)/1e2, '--', label=f'Explícito ({rotulos[idx]})')
    ax1e.plot(x_num, (res_imp_pp[dia] - p_ana_n)/1e2, label=f'Implícito ({rotulos[idx]})')

ax1.set_title('Validação P-P: Numérico vs Analítico')
ax1.set_xlabel('Distância x (m)')
ax1.set_ylabel('Pressão (bar)')
ax1.grid(True)
ax1.legend(loc='lower right')

ax1e.set_title('Erro em relação à solução analítica')
ax1e.set_xlabel('Distância x (m)')
ax1e.set_ylabel('$p_{num} - p_{ana}$ (mbar)')
ax1e.grid(True)
ax1e.legend(fontsize=8, ncol=2)

#grafico 2: vazão pressão
fig2, (ax2, ax2e) = plt.subplots(1, 2, figsize=(13, 5))
for idx, dia in enumerate(tempos_dias):
    t_seg = dia * DIA
    p_ana = analitica_qp(x_ana, t_seg, eta, L, p0, qw, mu, k, A)
    ax2.plot(x_ana, p_ana/1e5, color='black', label='Analítica' if idx == 0 else "")
    ax2.scatter(x_num, res_exp_qp[dia]/1e5, marker=marcadores[idx], facecolors='none',
                edgecolors='r', s=40, label='Explícito' if idx == 0 else "")
    ax2.scatter(x_num, res_imp_qp[dia]/1e5, marker='x', color='b', s=40,
                label='Implícito' if idx == 0 else "")
    ax2.annotate(rotulos[idx], (x_ana[20], p_ana[20]/1e5), fontsize=10,
                 textcoords='offset points', xytext=(8, -10))

    p_ana_n = analitica_qp(x_num, t_seg, eta, L, p0, qw, mu, k, A)
    ax2e.plot(x_num, (res_exp_qp[dia] - p_ana_n)/1e2, '--', label=f'Explícito ({rotulos[idx]})')
    ax2e.plot(x_num, (res_imp_qp[dia] - p_ana_n)/1e2, label=f'Implícito ({rotulos[idx]})')

r_inv = np.sqrt(4 * eta * max(tempos_dias) * DIA)
ax2.axvline(r_inv, color='gray', ls=':', lw=1.5)
ax2.annotate(f'$\\sqrt{{4\\eta t}}$ = {r_inv:.0f} m\n({rotulos[-1]})',
             (r_inv, 197), fontsize=8, color='gray',
             textcoords='offset points', xytext=(6, 0))
ax2.set_title('Validação Q-P: Numérico vs Analítico')
ax2.set_xlabel('Distância x (m)')
ax2.set_ylabel('Pressão (bar)')
ax2.grid(True)
ax2.legend(loc='lower right')

ax2e.set_title('Erro em relação à solução analítica')
ax2e.set_xlabel('Distância x (m)')
ax2e.set_ylabel('$p_{num} - p_{ana}$ (mbar)')
ax2e.grid(True)
ax2e.legend(fontsize=8, ncol=2)

plt.show()

