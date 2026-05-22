# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:23:33 2026

@author: BIEL
"""

# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np  

################## Preprocessamento ################## 

# Arquivo separado

################## Regressão com Árvores de Decisão ################## 



_ALTURA = 5

from sklearn.tree import DecisionTreeRegressor
_regressor = DecisionTreeRegressor(
        max_depth = _ALTURA,
        random_state = 0
    )

#  Treinamento
_regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = _regressor.predict(previsores_teste)





################## Avaliação dos resultados ################## 

_score = _regressor.score(previsores_teste, objetivo_teste)

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





# =============================================================================
#                              Exportando a Árvore
# =============================================================================

# from sklearn.tree import export_graphviz
# export_graphviz(_regressor,out_file="tree.dot",
#                 feature_names=_cols_previsores, 
#                 impurity=False, filled=True)

# # Visualizando a árvore
# import graphviz
# with open("tree.dot") as f:
#     dot_graph = f.read()
# display(graphviz.Source(dot_graph))

# # Visualizando a importância das características
# import matplotlib.pyplot as plt
# import numpy as np
# n_features = previsores.columns.size
# plt.barh(range(n_features), _regressor.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), previsores.columns)
# plt.xlabel("Feature importance")
# plt.ylabel("Feature")


