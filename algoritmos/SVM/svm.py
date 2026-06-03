# -*- coding: utf-8 -*-

from sklearn.svm import SVR
from sklearn import metrics
# =============================================================================
# Treinamento e Avaliação do SVR
# =============================================================================

# Inicialização do modelo SVR
regressor = SVR(kernel='linear', C=1.0, gamma='scale', epsilon=0.1)

# Treinamento do modelo
regressor.fit(previsores_treinamento, objetivo_treinamento.ravel())

# Predição nos dados de teste
previsoes_padrao = regressor.predict(previsores_teste)

# Inversão da padronização para escala original (Dólares)
previsoes_reais = _scaler_objetivo.inverse_transform(previsoes_padrao.reshape(-1, 1))
objetivo_teste_real = _scaler_objetivo.inverse_transform(objetivo_teste)

# Cálculo das métricas de desempenho
score = metrics.r2_score(objetivo_teste_real, previsoes_reais)
mae = metrics.mean_absolute_error(objetivo_teste_real, previsoes_reais)
mse = metrics.mean_squared_error(objetivo_teste_real, previsoes_reais)
rmse = np.sqrt(mse)

def br(v, c=2):
    return f"{v:,.{c}f}".replace(",", "_").replace(".", ",").replace("_", ".")
# Exibição dos resultados no console
print('\n=== DESEMPENHO DO SVR ===')
print(f'R² Score: {br(score, 4)}')  
print(f'MAE: R$ {br(mae)}')  
print(f'MSE: {br(mse)}')
print(f'RMSE: {br(rmse)}')
