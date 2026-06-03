# -*- coding: utf-8 -*-
from sklearn import metrics

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes=(50, 9),  
                           alpha=10.0,  
                           activation='relu',  
                           max_iter=300
 )

#  Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

previsoes_escala_original = _scaler_objetivo.inverse_transform(previsoes.reshape(-1, 1))
objetivo_escala_original = _scaler_objetivo.inverse_transform(objetivo_teste.reshape(-1, 1))

########### Avaliação dos resultados ###############

score = regressor.score(previsores_teste, objetivo_teste)

mae = metrics.mean_absolute_error(objetivo_escala_original, previsoes_escala_original)
mse = metrics.mean_squared_error(objetivo_escala_original, previsoes_escala_original)
rmse = np.sqrt(mse)

def br(v, c=2):
    return f"{v:,.{c}f}".replace(",", "_").replace(".", ",").replace("_", ".")

# Exibição dos resultados da Rede Neural no console
print('\n=== DESEMPENHO DA REDE NEURAL (MLP) ===')
print(f'R² Score: {br(score, 4)}')  
print(f'Mean Absolute Error (MAE): R$ {br(mae)}')  
print(f'Mean Squared Error (MSE): {br(mse)}')
print(f'Root Mean Squared Error (RMSE): R$ {br(rmse)}')

_colunas_previsores = previsores.columns

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 25))
plt.imshow(regressor.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(len(_colunas_previsores)), _colunas_previsores)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
