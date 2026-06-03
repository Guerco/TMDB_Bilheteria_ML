# -*- coding: utf-8 -*-
"""
Validação Cruzada - Random Forest Regressor

@author: Yuri Viana
"""

from pre_processamento import previsores, objetivo

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ==================================================
# CONFIGURAÇÕES
# ==================================================

_VC_PADRONIZACAO = False
_VC_PADRONIZACAO_OBJETIVO = False

# ==================================================
# FUNÇÃO WAPE
# ==================================================

def weighted_absolute_percentage_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

# ==================================================
# PREPARAÇÃO DOS DADOS
# ==================================================

if not isinstance(previsores, np.ndarray):
    previsores = previsores.values

if not isinstance(objetivo, np.ndarray):
    objetivo = objetivo.values.ravel()

# ==================================================
# VALIDAÇÃO CRUZADA
# ==================================================

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=3
)

scores = []
maes = []
mses = []
rmses = []
wapes = []

print("Iniciando treinamento...")

for indice_treinamento, indice_teste in kfold.split(previsores):

    X_treino = previsores[indice_treinamento]
    X_teste = previsores[indice_teste]

    y_treino = objetivo[indice_treinamento]
    y_teste = objetivo[indice_teste]

    # ==============================================
    # PADRONIZAÇÃO
    # ==============================================

    if _VC_PADRONIZACAO:
        scaler_x = StandardScaler()

        X_treino_scaled = scaler_x.fit_transform(X_treino)
        X_teste_scaled = scaler_x.transform(X_teste)

    if _VC_PADRONIZACAO_OBJETIVO:
        scaler_y = StandardScaler()

        y_treino_scaled = scaler_y.fit_transform(
            y_treino.reshape(-1, 1)
        ).ravel()

    # ==============================================
    # RANDOM FOREST
    # ==============================================

    regressor = RandomForestRegressor(
        n_estimators=100,
        max_features=10,
        max_depth=15,
        random_state=0,
        n_jobs=-1
    )

    if _VC_PADRONIZACAO and _VC_PADRONIZACAO_OBJETIVO:
        regressor.fit(
            X_treino_scaled,
            y_treino_scaled
        )

    elif _VC_PADRONIZACAO:
        regressor.fit(
            X_treino_scaled,
            y_treino
        )

    elif _VC_PADRONIZACAO_OBJETIVO:
        regressor.fit(
            X_treino,
            y_treino_scaled
        )

    else:
        regressor.fit(
            X_treino,
            y_treino
        )

    # ==============================================
    # PREVISÕES
    # ==============================================

    if _VC_PADRONIZACAO_OBJETIVO:

        if _VC_PADRONIZACAO:
            previsoes_scaled = regressor.predict(X_teste_scaled)
        else:
            previsoes_scaled = regressor.predict(X_teste)

        previsoes = scaler_y.inverse_transform(
            previsoes_scaled.reshape(-1, 1)
        ).ravel()

    else:

        if _VC_PADRONIZACAO:
            previsoes = regressor.predict(X_teste_scaled)
        else:
            previsoes = regressor.predict(X_teste)

    # ==============================================
    # MÉTRICAS
    # ==============================================

    score = metrics.r2_score(y_teste, previsoes)
    mae = metrics.mean_absolute_error(y_teste, previsoes)
    mse = metrics.mean_squared_error(y_teste, previsoes)
    rmse = np.sqrt(mse)
    wape = weighted_absolute_percentage_error(
        y_teste,
        previsoes
    )

    scores.append(score)
    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)
    wapes.append(wape)

# ==================================================
# RESULTADO FINAL
# ==================================================

print("\n===== RANDOM FOREST + VALIDAÇÃO CRUZADA =====\n")

print(f"R² Médio: {np.mean(scores):.5f}")
print(f"R² Desvio: {np.std(scores):.5f}")

print()

print(f"WAPE Médio: {np.mean(wapes):.5f}")
print(f"WAPE Desvio: {np.std(wapes):.5f}")

print()

print(f"MAE Médio: {np.mean(maes):.2f}")
print(f"MAE Desvio: {np.std(maes):.2f}")

print()

print(f"RMSE Médio: {np.mean(rmses):.2f}")
print(f"RMSE Desvio: {np.std(rmses):.2f}")

print()

print(f"MSE Médio: {np.mean(mses):.2f}")
print(f"MSE Desvio: {np.std(mses):.2f}")

# ==================================================
# GRÁFICOS
# ==================================================

sns.set_style("whitegrid")

# Último fold

if _VC_PADRONIZACAO:

    previsoes_treinamento = regressor.predict(
        X_treino_scaled
    )

else:

    previsoes_treinamento = regressor.predict(
        X_treino
    )

if _VC_PADRONIZACAO_OBJETIVO:

    previsoes_treinamento = scaler_y.inverse_transform(
        previsoes_treinamento.reshape(-1, 1)
    ).ravel()

# ==================================================
# ERROS RELATIVOS
# ==================================================

epsilon = 1e-10

erros_treinamento = (
    y_treino - previsoes_treinamento
) / (np.abs(y_treino) + epsilon)

erros_teste = (
    y_teste - previsoes
) / (np.abs(y_teste) + epsilon)

# ==================================================
# 1 - RESÍDUOS
# ==================================================

plt.figure(figsize=(10,6))

sns.residplot(
    x=y_treino,
    y=previsoes_treinamento,
    lowess=False,
    label='Treinamento'
)

sns.residplot(
    x=y_teste,
    y=previsoes,
    lowess=False,
    label='Teste'
)

plt.title('Gráfico de Resíduos')
plt.xlabel('Valor Real')
plt.ylabel('Resíduo')
plt.legend()

# ==================================================
# 2 - PREVISÃO VS REAL
# ==================================================

plt.figure(figsize=(10,6))

plt.scatter(
    y_treino,
    previsoes_treinamento,
    alpha=0.5,
    label='Treinamento'
)

plt.scatter(
    y_teste,
    previsoes,
    alpha=0.5,
    label='Teste'
)

min_val = min(y_treino.min(), y_teste.min())
max_val = max(y_treino.max(), y_teste.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    'r--',
    linewidth=2,
    label='Previsão Perfeita'
)

plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Previsão vs Real')
plt.legend()

# ==================================================
# 3 - DESVIO RELATIVO
# ==================================================

plt.figure(figsize=(10,6))

sns.histplot(
    erros_treinamento,
    kde=True,
    stat='density',
    alpha=0.5,
    label='Treinamento'
)

sns.histplot(
    erros_teste,
    kde=True,
    stat='density',
    alpha=0.5,
    label='Teste'
)

plt.xlabel('Desvio Relativo')
plt.ylabel('Densidade')
plt.title('Distribuição do Desvio Relativo')
plt.legend()

plt.show()