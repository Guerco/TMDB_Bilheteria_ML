# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:35:02 2026

@author: BIEL
"""

# =============================================================================

# Configure aqui as padronizações que devem ser utilizadas 
_VC_PADRONIZACAO          = False
_VC_PADRONIZACAO_OBJETIVO = False

# =============================================================================





import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# =============================================================================

# Função WAPE
def weighted_absolute_percentage_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))





# =============================================================================
#                              Validação Cruzada
# =============================================================================

# Divisão dos dados para validação cruzada (KFold ideal para regressão)
kfold = KFold(n_splits=5, shuffle=True, random_state=3)

scores = []
maes = []
mses = []
rmses = []
wapes = []

if not isinstance(previsores, np.ndarray):
    previsores = previsores.values
if not isinstance(objetivo, np.ndarray):
    objetivo = objetivo.values.ravel()

print("Iniciando treinamento...")

for indice_treinamento, indice_teste in kfold.split(previsores):
    
    # 1. Separar os dados PRIMEIRO (na escala original)
    X_treino = previsores[indice_treinamento]
    X_teste = previsores[indice_teste]
    
    y_treino = objetivo[indice_treinamento]
    y_teste = objetivo[indice_teste]
    
    # 2. Padronização DENTRO do loop (evita Data Leakage)
    if _VC_PADRONIZACAO:
        _scaler_x = StandardScaler()
        X_treino_scaled = _scaler_x.fit_transform(X_treino)
        X_teste_scaled = _scaler_x.transform(X_teste)
    
    if _VC_PADRONIZACAO_OBJETIVO:
        _scaler_y = StandardScaler()
        y_treino_scaled = _scaler_y.fit_transform(y_treino.reshape(-1, 1)).ravel()
    
    # 3. Construção e Treinamento do Modelo

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    _poly = PolynomialFeatures(degree=2)

    if _VC_PADRONIZACAO:
        X_treino_scaled = _poly.fit_transform(X_treino_scaled)
        X_teste_scaled = _poly.transform(X_teste_scaled)
    else:
        X_treino = _poly.fit_transform(X_treino)
        X_teste = _poly.transform(X_teste)
    
    regressor = LinearRegression()
    
    # Treinamento
    if _VC_PADRONIZACAO and _VC_PADRONIZACAO_OBJETIVO:
        regressor.fit(X_treino_scaled, y_treino_scaled.ravel())
    elif _VC_PADRONIZACAO:
        regressor.fit(X_treino_scaled, y_treino.ravel())
    elif _VC_PADRONIZACAO_OBJETIVO:
        regressor.fit(X_treino, y_treino_scaled.ravel())
    else:
        regressor.fit(X_treino, y_treino.ravel())
    
    # 4. Previsões
    if _VC_PADRONIZACAO_OBJETIVO:
        if _VC_PADRONIZACAO:
            previsoes_scaled = regressor.predict(X_teste_scaled)
        else:
            previsoes_scaled = regressor.predict(X_teste)
        previsoes = _scaler_y.inverse_transform(previsoes_scaled.reshape(-1, 1)).ravel()
    else:
        if _VC_PADRONIZACAO:
            previsoes = regressor.predict(X_teste_scaled)
        else:
            previsoes = regressor.predict(X_teste)
    
    # 5. Avaliação
    score = metrics.r2_score(y_teste, previsoes)
    mae = metrics.mean_absolute_error(y_teste, previsoes)
    mse = metrics.mean_squared_error(y_teste, previsoes)
    rmse = np.sqrt(mse)
    wape = weighted_absolute_percentage_error(y_teste, previsoes)

    scores.append(score)
    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)
    wapes.append(wape)


######################## Resultado final ########################
# Métricas médias
scores = np.asarray(scores)
score_final_medio = scores.mean()
score_final_desvio_padrao = scores.std()

maes = np.asarray(maes)
mae_final_medio = maes.mean()
mae_final_desvio_padrao = maes.std()

mses = np.asarray(mses)
mse_final_medio = mses.mean()
mse_final_desvio_padrao = mses.std()

rmses = np.asarray(rmses)
rmse_final_medio = rmses.mean()
rmse_final_desvio_padrao = rmses.std()

wapes = np.asarray(wapes)
wape_final_medio = wapes.mean()
wape_final_desvio_padrao = wapes.std()

print("\n--- Resultados Finais ---")

print("\n    - Médios -")
print(f"R²Score Médio: {score_final_medio:.5f}".replace('.', ','))
print(f"WAPE Médio: {wape_final_medio:.5f}".replace('.', ','))
print(f"MAE Médio: {mae_final_medio:.2f}".replace('.', ','))
print(f"RMSE Médio: {rmse_final_medio:.5f}".replace('.', ','))
print(f"MSE Médio: {mse_final_medio:.5f}".replace('.', ','))

print("\n\n    - Desvios Padrão -")
print(f"R²Score Desvio Padrão: {score_final_desvio_padrao:.5f}".replace('.', ','))
print(f"WAPE Desvio Padrão: {wape_final_desvio_padrao:.5f}".replace('.', ','))
print(f"MAE Desvio Padrão: {mae_final_desvio_padrao:.2f}".replace('.', ','))
print(f"RMSE Desvio Padrão: {rmse_final_desvio_padrao:.5f}".replace('.', ','))
print(f"MSE Desvio Padrão: {mse_final_desvio_padrao:.5f}".replace('.', ','))





# =============================================================================
#                          Gráficos de Avaliação
# =============================================================================

sns.set_style("whitegrid")
sns.despine(top=True, right=False, left=False, bottom=False, offset=None, trim=False)

# Usando o modelo da última iteração (último fold) para plotar os gráficos
if _VC_PADRONIZACAO:
    previsoes_treinamento_scaled = regressor.predict(X_treino_scaled)
    # previsoes_treinamento = _scaler_y.inverse_transform(previsoes_treinamento_scaled.reshape(-1, 1))
else:
    previsoes_treinamento = regressor.predict(X_treino)


# Cálculo dos erros (desvio relativo)
erros_treinamento = (y_treino - previsoes_treinamento) / y_treino
erros_teste = (y_teste - previsoes) / y_teste

# 1. Gráfico de Resíduos (Residplot)
plt.figure(figsize=(8, 5))
# .ravel() é usado para transformar a matriz 2D em 1D e evitar warnings do seaborn
ax1 = sns.residplot(x=y_treino.ravel(), y=previsoes_treinamento.ravel(), lowess=False, color="blue", label='Treinamento')
ax1 = sns.residplot(x=y_teste.ravel(), y=previsoes.ravel(), lowess=False, color="orange", label='Teste')
ax1.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax1.set_xlabel("Valor Real", fontsize=12)
ax1.set_ylabel("Resíduos", fontsize=12)
ax1.set_title("Gráfico de Resíduos")

# 2. Gráfico de Previsão vs Real
plt.figure(figsize=(8, 5))
plt.scatter(x=y_treino, y=previsoes_treinamento, alpha=0.5, label='Treinamento', color="blue")
plt.scatter(x=y_teste, y=previsoes, alpha=0.5, label='Teste', color="orange")
# Adicionando uma reta de referência (onde Previsão = Valor Real)
min_val = min(y_treino.min(), y_teste.min())
max_val = max(y_treino.max(), y_teste.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Previsão Perfeita')
plt.xlabel("Valor Real")
plt.ylabel("Previsão")
plt.title("Previsões vs Valores Reais")
plt.legend()

# 3. Histograma dos resíduos (Desvio Relativo)
plt.figure(figsize=(8, 5))
# Atualizado de sns.distplot para sns.histplot 
ax2 = sns.histplot(erros_treinamento.ravel(), kde=True, stat="density", color="blue", label="Treinamento", alpha=0.4)
ax2 = sns.histplot(erros_teste.ravel(), kde=True, stat="density", color="orange", label="Teste", alpha=0.4)
ax2.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax2.set_xlabel("Desvio Relativo", fontsize=12)
ax2.set_ylabel("Densidade", fontsize=12)
ax2.set_title("Distribuição do Desvio Relativo")

plt.show()