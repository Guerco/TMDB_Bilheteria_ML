# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:49:08 2026

@author: Yuri Viana
"""

################## Preprocessamento ##################

from pre_processamento import *

################## Bibliotecas ##################

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd

################## Regressão com Random Forest ##################

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(
    n_estimators=100,      # número de árvores
    max_features=10,       # quantidade de atributos por divisão
    max_depth=15,
    random_state=0,
    n_jobs=-1
)

################## Treinamento ##################

regressor.fit(
    previsores_treinamento,
    objetivo_treinamento.values.ravel()
)

################## Teste ##################

previsoes = regressor.predict(previsores_teste)

################## Avaliação ##################

score = regressor.score(
    previsores_teste,
    objetivo_teste
)

mae = metrics.mean_absolute_error(
    objetivo_teste,
    previsoes
)

mse = metrics.mean_squared_error(
    objetivo_teste,
    previsoes
)

rmse = np.sqrt(mse)

print('\n===== RANDOM FOREST REGRESSOR =====\n')

print('R² Score:', score)
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)

################## Importância das variáveis ##################

n_features = previsores.columns.size

plt.figure(figsize=(10,8))

plt.barh(
    range(n_features),
    regressor.feature_importances_
)

plt.yticks(
    np.arange(n_features),
    previsores.columns
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Importância das Variáveis - Random Forest")

plt.tight_layout()
plt.show()

