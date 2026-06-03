import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import numpy as np

training_score = []
test_score = []

# Testando o alpha (freio) saltando em potências de 10
valores_alpha = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

for a in valores_alpha:
    # Rede fixada em 50 neurônios, variando apenas o alpha
    regressor = MLPRegressor(activation='relu',
                             max_iter=300,
                             verbose=False,
                             hidden_layer_sizes=(50, 9), 
                             alpha=a,  
                             random_state=0)
    
    regressor.fit(previsores_treinamento, np.ravel(objetivo_treinamento))
    
    training_score.append(regressor.score(previsores_treinamento, objetivo_treinamento))
    test_score.append(regressor.score(previsores_teste, objetivo_teste))

plt.figure(figsize=(10, 6))

# Usamos semilogx em vez de plot normal porque os valores de alpha dão saltos muito grandes
plt.semilogx(valores_alpha, training_score, label="Treinamento", marker='o', color='royalblue')
plt.semilogx(valores_alpha, test_score, label="Teste", marker='o', color='darkorange')

plt.ylabel("R² Score (Desempenho)")
plt.xlabel("Taxa de Regularização (Alpha) - Escala Logarítmica")
plt.title("Curva de Regularização: Controlando o Overfitting")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()