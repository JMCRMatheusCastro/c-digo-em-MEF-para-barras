import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Universidade Federal de Pernambuco (UFPE)
# José Matheus de Castro Rodrigues
# Data: 29/11/2024

# OTIMIZAÇÃO DO CUSTO DE UM PILAR EM CONCRETO ARMADO
# SOB AS VARIÁVEIS BASE E ÁREA DE AÇO

# Dados de entrada:
h = 40           # Altura (cm)
d_linha = 4      # Distância do centro das camadas 1 e nLinha até as bordas (cm)
n_linha = 2      # Número de camadas da armadura
n = np.array([2, 2])  # Número de barras em cada camada

fck = 2          # Resistência característica à compressão do concreto (kN/cm²)
fyk = 50         # Tensão de escoamento característica do aço (kN/cm²)
Es = 21000       # Módulo de elasticidade do aço (kN/cm²)
densidade_aco = 7850  # Densidade do aço (kg/m³)

Nk = 410         # Esforço normal de serviço (kN)
Mk = Nk * 25     # Momento fletor de serviço (kN.cm)

preco_aco = 9.26        # Preço do aço (R$/kg)
preco_concreto = 607.21 # Preço do concreto (R$/m³)

# Resistências de projeto:
fcd = fck / 1.4
Sigmacd = 0.85 * fcd
fyd = fyk / 1.15

# Esforços de projeto:
Nd = 1.4 * Nk
Md = 1.4 * Mk

# Parâmetros geométricos:
delta = d_linha / h
Beta = delta + ((n_linha - np.arange(1, n_linha + 1)) * (1 - 2 * delta) / (n_linha - 1))
d = Beta * h

# Deformações limites:
epsilon_0 = 0.002
epsilon_u = 0.0035
k = 1 - (epsilon_0 / epsilon_u)

# Funções auxiliares:
def calcular_xi(x, h):
    return x / h

def calcular_deformacoes_aco(epsilon_0, xi, Beta, k):
    return epsilon_0 * ((xi - Beta) / (k - xi))

def calcular_tensao_aco(epsilon_s, Es, fyd):
    epsilon_yd = fyd / Es
    return np.where(np.abs(epsilon_s) <= epsilon_yd, Es * epsilon_s, np.sign(epsilon_s) * fyd)

def calcular_nu_mi(Nd, Md, b, h, Sigmacd):
    nu = Nd / (b * h * Sigmacd)
    mi = Md / (b * h**2 * Sigmacd)
    return nu, mi

def calcular_rc_betac(xi):
    rc = min(0.8 * xi, 1)
    Betac = min(0.5 * 0.8 * xi, 0.5)
    return rc, Betac

def calcular_ast(vars, h, fyd, Nd, Md, Sigmacd, epsilon_0, k, n, Beta, Es):
    x, b = vars
    xi = calcular_xi(x, h)
    epsilon_s = calcular_deformacoes_aco(epsilon_0, xi, Beta, k)
    sigma_sd = calcular_tensao_aco(epsilon_s, Es, fyd)
    nu, mi = calcular_nu_mi(Nd, Md, b, h, Sigmacd)
    rc, Betac = calcular_rc_betac(xi)
    SOM1 = np.sum(n * sigma_sd)
    SOM2 = np.sum(n * sigma_sd * Beta)
    omega = (mi - (0.5 * nu) + rc * Betac) * SOM1 + (nu - rc) * SOM2
    return np.abs(omega * b * h * Sigmacd / fyd)

def custo(vars, h, fyd, Nd, Md, Sigmacd, epsilon_0, k, n, Beta, Es, densidade_aco, preco_aco, preco_concreto):
    x, b = vars
    Ast = calcular_ast(vars, h, fyd, Nd, Md, Sigmacd, epsilon_0, k, n, Beta, Es)
    custo_aco = Ast * (1 / 10000) * densidade_aco * preco_aco
    custo_concreto = b * h * (1 / 10000) * preco_concreto
    return custo_aco + custo_concreto

def restricoes(vars, h, fyd, Nd, Md, Sigmacd, epsilon_0, k, n, Beta, Es):
    x, b = vars
    Ast = calcular_ast(vars, h, fyd, Nd, Md, Sigmacd, epsilon_0, k, n, Beta, Es)
    g1 = 0.004 * b * h - Ast
    g2 = Ast - 0.08 * b * h
    return [g1, g2]

# Configuração da otimização:
chute_inicial = [h / 2, 20]  # [x, b]
limites = [(0, None), (19, None)]

resultado = minimize(
    fun=custo,
    x0=chute_inicial,
    args=(h, fyd, Nd, Md, Sigmacd, epsilon_0, k, n, Beta, Es, densidade_aco, preco_aco, preco_concreto),
    method='SLSQP',
    bounds=limites,
    constraints={'type': 'ineq', 'fun': restricoes, 'args': (h, fyd, Nd, Md, Sigmacd, epsilon_0, k, n, Beta, Es)}
)

# Resultados:
if resultado.success:
    print("Otimização bem-sucedida!")
    print(f"Dimensões otimizadas (bxh): {resultado.x[1]:.2f} cm x {h:.2f} cm")
    print(f"Profundidade da linha neutra: {resultado.x[0]:.2f} cm")
    print(f"Custo total: R$ {resultado.fun:.2f}/m")
else:
    print("Otimização falhou!")
    print(resultado.message)

# Visualização do histórico (se necessário):
plt.figure()
plt.title("Histórico de Convergência")
plt.plot(resultado.nit, resultado.fun, '-o')
plt.xlabel("Iteração")
plt.ylabel("Custo (R$)")
plt.grid()
plt.show()
