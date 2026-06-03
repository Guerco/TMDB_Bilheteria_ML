# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn import metrics

# Salva os nomes das colunas ANTES da padronização virar array numpy
if isinstance(previsores, pd.DataFrame):
    nomes_colunas = previsores.columns.tolist()
else:
    nomes_colunas = [f'Variável {i}' for i in range(previsores.shape[1])]

# Inicialização e treinamento
regressor = SVR(kernel='linear', C=1.0, gamma='scale', epsilon=0.1)
regressor.fit(previsores_treinamento, objetivo_treinamento.ravel())

# Predição e inversão de escala
previsoes_padrao = regressor.predict(previsores_teste)
previsoes_reais = _scaler_objetivo.inverse_transform(previsoes_padrao.reshape(-1, 1))
objetivo_teste_real = _scaler_objetivo.inverse_transform(objetivo_teste)

# Métricas
score = metrics.r2_score(objetivo_teste_real, previsoes_reais)
mae = metrics.mean_absolute_error(objetivo_teste_real, previsoes_reais)
mse = metrics.mean_squared_error(objetivo_teste_real, previsoes_reais)
rmse = np.sqrt(mse)

def br(v, c=2):
    return f"{v:,.{c}f}".replace(",", "_").replace(".", ",").replace("_", ".")

print('\n=== DESEMPENHO DO SVR ===')
print(f'R² Score: {br(score, 4)}')
print(f'MAE: R$ {br(mae)}')
print(f'MSE: {br(mse)}')
print(f'RMSE: {br(rmse)}')

# =============================================================================
# Importância das Variáveis (Top 15)
# =============================================================================
importancias = np.abs(regressor.coef_.ravel())

df_importancia = pd.DataFrame({
    'Variável': nomes_colunas,
    'Relevância Absoluta': importancias
}).sort_values(by='Relevância Absoluta', ascending=False)

df_top_15 = df_importancia.head(15)

sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x='Relevância Absoluta',
    y='Variável',
    data=df_top_15,
    palette='viridis'
)

offset = df_top_15['Relevância Absoluta'].max() * 0.01
for i, v in enumerate(df_top_15['Relevância Absoluta']):
    ax.text(v + offset, i, f"{v:.4f}", va='center', fontsize=10, fontweight='bold')

plt.title('Top 15 Variáveis Mais Relevantes no Modelo SVR', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Magnitude do Coeficiente (Peso Absoluto)', fontsize=12)
plt.ylabel('Variáveis Preditoras', fontsize=12)
plt.xlim(0, df_top_15['Relevância Absoluta'].max() * 1.12)
plt.tight_layout()
plt.show()