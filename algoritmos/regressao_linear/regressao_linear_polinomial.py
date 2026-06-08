# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:01:02 2026

@author: BIEL
"""

# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  

################## Preprocessamento ################## 

# Arquivo separado

################## Regressão Polinomial ################## 

_DEGREE = 3

from sklearn.preprocessing import PolynomialFeatures
_poly = PolynomialFeatures(degree = _DEGREE)
previsores_treinamento_poly = _poly.fit_transform(previsores_treinamento)
previsores_teste_poly = _poly.transform(previsores_teste)

from sklearn.linear_model import LinearRegression
_regressor = LinearRegression()

#  Treinamento
_regressor.fit(previsores_treinamento_poly, objetivo_treinamento)

score_treinamento = _regressor.score(previsores_treinamento_poly, objetivo_treinamento)

# Teste
previsoes = _regressor.predict(previsores_teste_poly)

################## Avaliação dos resultados ################## 

_score = _regressor.score(previsores_teste_poly, objetivo_teste)

if _PADRONIZACAO_OBJETIVO:
    previsoes = _scaler_objetivo.inverse_transform(previsoes.reshape(-1, 1))
    objetivo_teste = _scaler_objetivo.inverse_transform(objetivo_teste.reshape(-1, 1))
    
_mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
_mse = metrics.mean_squared_error(objetivo_teste, previsoes)
_rmse = np.sqrt(metrics.mean_squared_error(objetivo_teste, previsoes))

print('Score:', f'{_score:.5f}'.replace('.', ','))  
print('Mean Absolute Error:', f'{_mae:.2f}'.replace('.', ','))  
print('Mean Squared Error:', f'{_mse:.5f}'.replace('.', ','))  
print('Root Mean Squared Error:', f'{_rmse:.2f}'.replace('.', ','))


# Parâmetros estimados para o modelo
coef_0 = _regressor.intercept_




