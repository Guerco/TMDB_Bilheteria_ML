# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:44:50 2026

@author: BIEL
"""

# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

training_accuracy = []
test_accuracy = []

# tentando diferentes valores de max_depth
neighbors_settings = range(1, 30)
for k in neighbors_settings:
    classificador = DecisionTreeRegressor(max_depth=k, random_state=0)
    classificador.fit(previsores_treinamento, objetivo_treinamento)    
    training_accuracy.append(classificador.score(previsores_treinamento, objetivo_treinamento))
    test_accuracy.append(classificador.score(previsores_teste, objetivo_teste))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Score")
plt.xlabel("Altura da árvore")
plt.legend()
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylim(0, 1)

plt.show()