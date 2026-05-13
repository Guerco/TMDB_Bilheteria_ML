# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:38:02 2026

@author: BIEL
"""

# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  

################## Preprocessamento ################## 

# Arquivo separado

################## Regressão Linear Múltipla ################## 

from sklearn.linear_model import LinearRegression
_regressor = LinearRegression()

#  Treinamento
_regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = _regressor.predict(previsores_teste)

################## Avaliação dos resultados ################## 

_score = _regressor.score(previsores_teste, objetivo_teste)
_mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
_mse = metrics.mean_squared_error(objetivo_teste, previsoes)
_rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', _score)  
print('Mean Absolute Error:', _mae)  
print('Mean Squared Error:', _mse)  
print('Root Mean Squared Error:', _rmse)

# Parâmetros estimados para o modelo
coef_0 = _regressor.intercept_
coeficientes = _regressor.coef_
