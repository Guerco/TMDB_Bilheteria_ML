# =============================================================================
# Busca de Hiperparâmetros Otimizada (Randomized Search)
# =============================================================================
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn import metrics

# Grade de parâmetros focada exclusivamente no kernel RBF
param_distributions = {
    'kernel': ['rbf'],
    'C': [0.1, 1.0, 10.0, 100.0],
    'epsilon': [0.01, 0.1, 0.2],
    'gamma': ['scale', 'auto']
}

svr_base = SVR()

# n_iter=6 define que o algoritmo testará apenas 6 combinações aleatórias
# cv=3 reduz o número de divisões da validação cruzada para acelerar a execução
random_search = RandomizedSearchCV(
    estimator=svr_base,
    param_distributions=param_distributions,
    n_iter=6,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    random_state=69,
    verbose=2
)

print("Iniciando busca otimizada (Total: 18 execuções)...")
random_search.fit(previsores_treinamento, objetivo_treinamento.ravel())

melhores_parametros = random_search.best_params_
melhor_r2_validacao = random_search.best_score_

print('\n=== CONFIGURAÇÃO ENCONTRADA ===')
print(f'Melhor R² Médio: {melhor_r2_validacao:.4f}')
print('Parâmetros:', melhores_parametros)

# =============================================================================
# Avaliação Final com o Melhor Modelo
# =============================================================================
melhor_regressor = random_search.best_estimator_

previsoes_padrao = melhor_regressor.predict(previsores_teste)

previsoes_reais = _scaler_objetivo.inverse_transform(previsoes_padrao.reshape(-1, 1))
objetivo_teste_real = _scaler_objetivo.inverse_transform(objetivo_teste)

score_final = metrics.r2_score(objetivo_teste_real, previsoes_reais)
mae_final = metrics.mean_absolute_error(objetivo_teste_real, previsoes_reais)

print('\n=== DESEMPENHO NO TESTE REAL ===')
print(f'R² Score Final: {score_final:.4f}')  
print(f'MAE Final: U$ {mae_final:,.2f}')