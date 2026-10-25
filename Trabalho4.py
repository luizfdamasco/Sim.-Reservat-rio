import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

MD = 0.9869233e-15
DIA = 86400.0

#dados de entrada
N, M = 25, 25
dx, dy = 20, 20
h = 10
phi = 0.2
mu = 1e-3
ct = 1e-9
p0 = 200e5
pwf = 100e5
rw = 0.1
dias_sim = 20
dt_dias = 0.25
dias_campo = [1, 3, 8, 20]   #instantes dos transientes

#permeabilidade heterogenea
np.random.seed(42)
k_base = 1e-13
k_mapa = k_base * np.random.lognormal(0, 0.8, (M, N))

# fluxo nulo nas fronteiras
cc_nulo = {'oeste': {'tipo': 'nulo', 'valor': 0.0},
           'leste': {'tipo': 'nulo', 'valor': 0.0},
           'sul':   {'tipo': 'nulo', 'valor': 0.0},
           'norte': {'tipo': 'nulo', 'valor': 0.0}}

#funcoes

def calc_coef_face(k1, k2, mu, d): # coeficiente da face do volume de controle
    if k1 == 0 or k2 == 0:
        return 0.0
    k_face = 2 * (k1 * k2) / (k1 + k2)
    return k_face / (mu * d**2)

def coef_contorno(cc, k_borda, mu, d):
    if cc['tipo'] == 'pressao':
        return 2.0 * k_borda / (mu * d**2)
    return 0.0

def fonte_contorno(cc, coef, dV):
    if cc['tipo'] == 'pressao':
        return coef * cc['valor']
    if cc['tipo'] == 'fluxo':
        return cc['valor'] / dV
    return 0.0

def monta_coeficientes(k_map, cc, N, M, dx, dy, mu):
    Ax = np.zeros((M, N+1))
    Ay = np.zeros((M+1, N))

    for j in range(M):
        for i in range(1, N):
            Ax[j,i] = calc_coef_face(k_map[j,i-1], k_map[j,i], mu, dx)
    for j in range(1, M):
        for i in range(N):
            Ay[j,i] = calc_coef_face(k_map[j-1,i], k_map[j,i], mu, dy)

    #faces das fronteiras
    for j in range(M):
        Ax[j,0] = coef_contorno(cc['oeste'], k_map[j,0], mu, dx)
        Ax[j,N] = coef_contorno(cc['leste'], k_map[j,N-1], mu, dx)
    for i in range(N):
        Ay[0,i] = coef_contorno(cc['sul'], k_map[0,i], mu, dy)
        Ay[M,i] = coef_contorno(cc['norte'], k_map[M-1,i], mu, dy)

    return Ax, Ay

def calc_velocidade(p_vetor, Ax, Ay, cc, N, M, dx, dy, mu):
    p_mat = p_vetor.reshape((M, N))
    vx = np.zeros((M, N))
    vy = np.zeros((M, N))

    for j in range(M):
        for i in range(N):
            # faces oeste e leste
            p_oeste = p_mat[j,i-1] if i > 0 else (cc['oeste']['valor']
                      if cc['oeste']['tipo'] == 'pressao' else p_mat[j,i])
            p_leste = p_mat[j,i+1] if i < N-1 else (cc['leste']['valor']
                      if cc['leste']['tipo'] == 'pressao' else p_mat[j,i])
            vx_oeste = Ax[j,i]   * (p_oeste - p_mat[j,i]) * dx
            vx_leste = Ax[j,i+1] * (p_mat[j,i] - p_leste) * dx
            vx[j,i] = 0.5 * (vx_oeste + vx_leste)

            # faces sul e norte
            p_sul = p_mat[j-1,i] if j > 0 else (cc['sul']['valor']
                    if cc['sul']['tipo'] == 'pressao' else p_mat[j,i])
            p_norte = p_mat[j+1,i] if j < M-1 else (cc['norte']['valor']
                      if cc['norte']['tipo'] == 'pressao' else p_mat[j,i])
            vy_sul = Ay[j,i]   * (p_sul - p_mat[j,i]) * dy
            vy_norte = Ay[j+1,i] * (p_mat[j,i] - p_norte) * dy
            vy[j,i] = 0.5 * (vy_sul + vy_norte)

    return vx, vy

def simulador_2d(k_map, pocos, cc, N, M, dx, dy, h, phi, mu, ct, p0, dias, dt_dias, dias_campo):
    n_volumes = N * M
    dt = dt_dias * DIA
    passos = int(round(dias / dt_dias))
    passos_campo = [int(round(d / dt_dias)) for d in dias_campo]

    # volume de controle e termo de acumulo
    dV = dx * dy * h
    acumulo = (phi * ct) / dt

    # raio equivalente e indice de poco
    req = 0.140365 * np.sqrt(dx**2 + dy**2)
    WI = (2 * np.pi * h) / np.log(req / rw)

    Ax, Ay = monta_coeficientes(k_map, cc, N, M, dx, dy, mu)

    A = lil_matrix((n_volumes, n_volumes))
    b_cc = np.zeros(n_volumes)

    for j in range(M):
        for i in range(N):
            p = j * N + i
            A_W, A_E = Ax[j,i], Ax[j,i+1]
            A_S, A_N = Ay[j,i], Ay[j+1,i]

            A[p, p] = acumulo + A_W + A_E + A_S + A_N
            if i > 0:   A[p, p-1] = -A_W
            if i < N-1: A[p, p+1] = -A_E
            if j > 0:   A[p, p-N] = -A_S
            if j < M-1: A[p, p+N] = -A_N

    # termo fonte das condicoes de contorno
    for j in range(M):
        b_cc[j*N]       += fonte_contorno(cc['oeste'], Ax[j,0], dV)
        b_cc[j*N + N-1] += fonte_contorno(cc['leste'], Ax[j,N], dV)
    for i in range(N):
        b_cc[i]           += fonte_contorno(cc['sul'], Ay[0,i], dV)
        b_cc[(M-1)*N + i] += fonte_contorno(cc['norte'], Ay[M,i], dV)

    # cpoços
    for poco in pocos:
        p_idx = poco['j'] * N + poco['i']
        lambd = k_map[poco['j'], poco['i']] / mu

        if poco['tipo'] == 'P':    # produtor com pressao prescrita
            A[p_idx, p_idx] += lambd * WI / dV
            b_cc[p_idx] += lambd * WI * poco['pwf'] / dV
        elif poco['tipo'] == 'Q':  # injetor com vazao prescrita
            b_cc[p_idx] += poco['qw'] / dV

    lu = splu(A.tocsc())

    p_old = np.ones(n_volumes) * p0
    vazao_prod = []
    prod_acumulada = []
    vol_acumulado = 0.0
    campos = {}

    for n in range(1, passos + 1):
        b = acumulo * p_old + b_cc
        p_new = lu.solve(b)

        # vazao nos produtores
        q_poco = 0.0
        for poco in pocos:
            if poco['tipo'] == 'P':
                p_idx = poco['j'] * N + poco['i']
                lambd = k_map[poco['j'], poco['i']] / mu
                q_poco += lambd * WI * (p_new[p_idx] - poco['pwf'])

        vazao_prod.append(q_poco * DIA)
        vol_acumulado += q_poco * dt
        prod_acumulada.append(vol_acumulado)

        #transientes de pressao e velocidade
        if n in passos_campo:
            vx, vy = calc_velocidade(p_new, Ax, Ay, cc, N, M, dx, dy, mu)
            campos[dias_campo[passos_campo.index(n)]] = (p_new.reshape((M, N)), vx, vy)

        p_old = p_new

    return campos, vazao_prod, prod_acumulada

#verificacao: caso unidimensional e atv 1

def analitica_pp(x, t, L, eta, pe, pw):
    soma = np.zeros(len(x))
    for n in range(1, 150):
        soma += (np.exp(-((n * np.pi / L)**2) * eta * t) / n) * np.sin(n * np.pi * x / L)
    return (pe - pw) * (x / L + (2 / np.pi) * soma) + pw

L_1d = 1000
N_1d, M_1d = 50, 5
dx_1d = L_1d / N_1d
dy_1d = 20
pw_1d = 150e5
pe_1d = 200e5
k_1d = np.ones((M_1d, N_1d)) * 1e-13
eta_1d = 1e-13 / (phi * mu * ct)

# pressao prescrita nas laterais e fluxo nulo norte e sul
cc_1d = {'oeste': {'tipo': 'pressao', 'valor': pw_1d},
         'leste': {'tipo': 'pressao', 'valor': pe_1d},
         'sul':   {'tipo': 'nulo', 'valor': 0.0},
         'norte': {'tipo': 'nulo', 'valor': 0.0}}

dias_1d = [0.25, 1.0, 2.0]
dt_1d = 0.002
campos_1d, _, _ = simulador_2d(k_1d, [], cc_1d, N_1d, M_1d, dx_1d, dy_1d, h,
                               phi, mu, ct, pe_1d, 2.0, dt_1d, dias_1d)

x_1d = np.linspace(dx_1d/2, L_1d - dx_1d/2, N_1d)
x_ana = np.linspace(0.1, L_1d, 200)

print('verificacao 1D (pressao prescrita nas laterais, fluxo nulo norte e sul)')
for d in dias_1d:
    p_num = campos_1d[d][0][M_1d//2, :]
    p_ana = analitica_pp(x_1d, d * DIA, L_1d, eta_1d, pe_1d, pw_1d)
    dif = (p_num - p_ana) / 1e5
    print('t =', d, 'dia(s) | erro max', round(np.abs(dif).max(), 4),
          'bar | erro rms', round(np.sqrt((dif**2).mean()), 4), 'bar')


#produtor no centro
pocos_c1 = [{'tipo': 'P', 'i': 12, 'j': 12, 'pwf': pwf}]
campos_c1, q_c1, Np_c1 = simulador_2d(k_mapa, pocos_c1, cc_nulo, N, M, dx, dy, h,
                                      phi, mu, ct, p0, dias_sim, dt_dias, dias_campo)

# produtor com falha selante
k_falha = np.copy(k_mapa)
k_falha[5:20, 16] = 0.0
pocos_c2 = [{'tipo': 'P', 'i': 8, 'j': 12, 'pwf': pwf}]
campos_c2, q_c2, Np_c2 = simulador_2d(k_falha, pocos_c2, cc_nulo, N, M, dx, dy, h,
                                      phi, mu, ct, p0, dias_sim, dt_dias, dias_campo)

#five-spot
q_inj_total = 0.002
pocos_c3 = [{'tipo': 'Q', 'i': 0,  'j': 0,  'qw': q_inj_total/4},
            {'tipo': 'Q', 'i': 24, 'j': 0,  'qw': q_inj_total/4},
            {'tipo': 'Q', 'i': 0,  'j': 24, 'qw': q_inj_total/4},
            {'tipo': 'Q', 'i': 24, 'j': 24, 'qw': q_inj_total/4},
            {'tipo': 'P', 'i': 12, 'j': 12, 'pwf': pwf}]
campos_c3, q_c3, Np_c3 = simulador_2d(k_mapa, pocos_c3, cc_nulo, N, M, dx, dy, h,
                                      phi, mu, ct, p0, dias_sim, dt_dias, dias_campo)


#balanco de massa
Vb = dx * dy * h * phi   # v poroso do bloco

print('\nbalanco de massa (t =', dias_sim, 'dias)')
for nome, campos, Np, k_map, q_inj in [('caso 1', campos_c1, Np_c1, k_mapa, 0.0),
                                       ('caso 2', campos_c2, Np_c2, k_falha, 0.0),
                                       ('caso 3', campos_c3, Np_c3, k_mapa, q_inj_total)]:
    P_fim = campos[dias_campo[-1]][0]
    mask = (k_map > 0)
    expansao = mask.sum() * Vb * ct * (p0 - P_fim[mask].mean())
    injetado = q_inj * dias_sim * DIA
    print(nome + ': Np =', round(Np[-1], 1), 'm3 | expansao + injecao =',
          round(expansao + injetado, 1), 'm3 | erro =',
          round(abs(Np[-1] - expansao - injetado) / (expansao + injetado) * 100, 4), '%')


# graficos

X, Y = np.meshgrid((np.arange(N) + 0.5)*dx, (np.arange(M) + 0.5)*dy)
tempos = np.arange(1, len(q_c1) + 1) * dt_dias
passo = 2

def plota_campos(fig_num, campos, titulo, k_map):
    selado = (k_map == 0)

    vx0, vy0 = campos[dias_campo[0]][1], campos[dias_campo[0]][2]
    escala = 12 * np.percentile(np.sqrt(vx0**2 + vy0**2), 90)

    fig, eixos = plt.subplots(2, len(dias_campo), figsize=(15, 7), num=fig_num)
    for idx, d in enumerate(dias_campo):
        P, vx, vy = campos[d]
        P = np.ma.masked_array(P, mask=selado)
        ax = eixos[0, idx]
        cf = ax.contourf(X, Y, P/1e5, levels=20, cmap='inferno')
        ax.set_title(str(d) + ' dia(s)')
        ax.set_xlabel('x (m)')
        if idx == 0: ax.set_ylabel('Pressão: y (m)')
        fig.colorbar(cf, ax=ax, label='bar' if idx == len(dias_campo)-1 else '')

        ax = eixos[1, idx]
        ax.quiver(X[::passo, ::passo], Y[::passo, ::passo],
                  vx[::passo, ::passo], vy[::passo, ::passo], scale=escala)
        ax.set_xlim(0, N*dx)
        ax.set_ylim(0, M*dy)
        ax.set_xlabel('x (m)')
        if idx == 0: ax.set_ylabel('Vel. Darcy: y (m)')
    fig.suptitle(titulo)
    fig.tight_layout()

# graficos 1 a 3: campos transientes de pressao e velocidade
plota_campos(1, campos_c1, 'Caso 1: produtor no centro', k_mapa)
plota_campos(2, campos_c2, 'Caso 2: produtor com falha selante', k_falha)
plota_campos(3, campos_c3, 'Caso 3: five-spot', k_mapa)

# grafico 4: curvas de producao e acumulado
plt.figure(4, figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(tempos, q_c1, label='Caso 1')
plt.plot(tempos, q_c2, label='Caso 2')
plt.plot(tempos, q_c3, label='Caso 3')
plt.title('Vazão de Produção')
plt.xlabel('Tempo (dias)')
plt.ylabel('m3/dia')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(tempos, Np_c1, label='Caso 1')
plt.plot(tempos, Np_c2, label='Caso 2')
plt.plot(tempos, Np_c3, label='Caso 3')
plt.title('Produção Acumulada')
plt.xlabel('Tempo (dias)')
plt.ylabel('m3')
plt.grid()
plt.legend()
plt.tight_layout()

# grafico 5: mapa de permeabilidade e falha
from matplotlib.colors import LogNorm
norma = LogNorm(vmin=k_mapa.min()/MD, vmax=k_mapa.max()/MD)

plt.figure(5, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.pcolormesh(X, Y, k_mapa/MD, cmap='jet', norm=norma)
plt.title('Perm. Heterogênea (mD)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.colorbar()

plt.subplot(1, 2, 2)
k_plot = np.copy(k_falha)
k_plot[k_plot == 0] = k_mapa.min()*1e-3
plt.pcolormesh(X, Y, k_plot/MD, cmap='jet', norm=norma)
plt.title('Perm. com Falha (mD)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.colorbar()
plt.tight_layout()

# grafico 6: verificacao 1D
plt.figure(6, figsize=(8, 5))
marcadores = ['o', 's', '^']
for idx, d in enumerate(dias_1d):
    p_ana = analitica_pp(x_ana, d * DIA, L_1d, eta_1d, pe_1d, pw_1d)
    plt.plot(x_ana, p_ana/1e5, color='black', label='Analítica' if idx == 0 else "")
    plt.scatter(x_1d, campos_1d[d][0][M_1d//2, :]/1e5, marker=marcadores[idx],
                facecolors='none', edgecolors='r', s=40,
                label='Simulador MVF' if idx == 0 else "")
    plt.annotate(str(d) + ' d', (x_ana[60], p_ana[60]/1e5), fontsize=10,
                 textcoords='offset points', xytext=(8, -4))

plt.title('Verificação: caso 1D e solução analítica da Atividade 1')
plt.xlabel('Distância x (m)')
plt.ylabel('Pressão (bar)')
plt.grid()
plt.legend()

plt.show()
