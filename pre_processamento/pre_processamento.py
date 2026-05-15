# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:10:40 2026

@author: BIEL
"""

# =============================================================================

# Configure aqui quais etapas do pré processamento deverão ser realizadas
_LABEL_ENCODER = True   # Não desativar label_encoder
_VARIAVEIS_DUMMY = True
_PADRONIZACAO = False

# =============================================================================





# =============================================================================

# Função Auxiliar para carregar e tratar a base na primeira execução
def carregarETratarBase():
    import os
    import pandas as pd
    
    # =============================================================================
    #                              Carregando a base
    # =============================================================================
    
    _diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    _caminho_base = os.path.join(_diretorio_atual, 'Filmes_TMDB.csv')
    
    base = pd.read_csv(_caminho_base, sep=',', encoding='latin-1')
    
    # print(base.columns.tolist())
    
    _colunas_selecionadas = [
        'vote_average',
        'vote_count',
        'status',
        'release_date',
        'revenue',
        'runtime',
        'adult',
        'budget',
        'original_language',
        # 'popularity',
        'genres',
        'production_countries'
    ]
    
    base = base[_colunas_selecionadas]
    
    # =============================================================================
    #                     Tratando valores inválidos
    # =============================================================================
    
    base_tratamento = base.copy()
    
    base_tratamento.drop(base_tratamento[base_tratamento['release_date'].isnull()].index, inplace=True)
    base_tratamento.drop(base_tratamento[base_tratamento['genres'].isnull()].index, inplace=True)
    base_tratamento.drop(base_tratamento[base_tratamento['production_countries'].isnull()].index, inplace=True)
    base_tratamento.drop(base_tratamento[base_tratamento['status'] != 'Released'].index, inplace=True)
    base_tratamento.drop(base_tratamento[base_tratamento['vote_count'] < 100].index, inplace=True)
    base_tratamento.drop(base_tratamento[base_tratamento['budget'] < 1000].index, inplace=True)
    base_tratamento.drop(base_tratamento[base_tratamento['runtime'] < 70].index, inplace=True)
    
    # =============================================================================
    #                     Separando dados em previsores e classe
    # =============================================================================
    
    # Convertendo data para mês e ano
    base_tratamento['release_date'] = pd.to_datetime(base_tratamento['release_date'])
    base_tratamento['release_year'] = base_tratamento['release_date'].dt.year
    base_tratamento['release_month'] = base_tratamento['release_date'].dt.month
    
    # Selecionando apenas os gêneros e países principais
    base_tratamento['genre_main'] = base_tratamento['genres'].str.split(', ').str[0]
    base_tratamento['country_main'] = base_tratamento['production_countries'].str.split(', ').str[0]
    
    # =============================================================================
    #                     Exportando base_tratamento para CSV
    # =============================================================================

    # Exportar para CSV no mesmo diretório
    base_tratamento.to_csv('./Filmes_TMDB_Tratada.csv', sep=',', encoding='utf-8', index=False)

    print("Base tratamento exportada com sucesso!")
    
    return base_tratamento

# =============================================================================





import os
import pandas as pd

_diretorio_atual = os.path.dirname(os.path.abspath(__file__))
_arquivo = os.path.join(_diretorio_atual, 'Filmes_TMDB_Tratada.csv')

_base_tratada_existe = os.path.exists(_arquivo)

if _base_tratada_existe:
    base_tratada = pd.read_csv(_arquivo, sep=',', encoding='latin-1')
else:
    base_tratada = carregarETratarBase()   
    

# =============================================================================
#                     Separando dados em previsores e classe
# =============================================================================

_cols_previsores = [
    'vote_average',
    # 'vote_count',
    # 'status',         # Campo utilizado apenas para filtrar
    # 'release_date',
    'release_year',
    'release_month',
    # 'revenue',
    'runtime',
    # 'adult',          # Sobraram apenas não adultos
    'budget',
    'original_language',
    # 'popularity',
    # 'genres',
    'genre_main',
    # 'production_countries',
    'country_main'
]

_cols_objetivo = [
    'revenue'
    # 'vote_average'
]

previsores = base_tratada[_cols_previsores].copy()
objetivo = base_tratada[_cols_objetivo].copy()

# Após separar previsores e classe, reseta o índice
previsores = previsores.reset_index(drop=True)
objetivo = objetivo.reset_index(drop=True)





# =============================================================================
#      LabelEncoder
# =============================================================================

from sklearn.preprocessing import LabelEncoder
import numpy as np

# _le_adult = LabelEncoder()
# previsores.loc[:, 'adult'] = _le_adult.fit_transform(previsores.loc[:, 'adult'])
# previsores['adult'] = previsores['adult'].astype('int64')

if _LABEL_ENCODER:
    _le_country_main = LabelEncoder()
    previsores.loc[:, 'country_main'] = _le_country_main.fit_transform(previsores.loc[:, 'country_main'])
    previsores['country_main'] = previsores['country_main'].astype('int64')
    
    if not _VARIAVEIS_DUMMY:
        _le_original_language = LabelEncoder()
        previsores.loc[:, 'original_language'] = _le_original_language.fit_transform(previsores.loc[:, 'original_language'])
        previsores['original_language'] = previsores['original_language'].astype('int64')
        
        _le_genre_main = LabelEncoder()
        previsores.loc[:, 'genre_main'] = _le_genre_main.fit_transform(previsores.loc[:, 'genre_main'])
        previsores['genre_main'] = previsores['genre_main'].astype('int64')        
        
        



# =============================================================================
#      Variáveis Dummy
# =============================================================================

if _VARIAVEIS_DUMMY:
    from sklearn.preprocessing import LabelBinarizer
    _lb = LabelBinarizer()

    # Variavel original_language
    _variaveis_dummy = _lb.fit_transform(previsores['original_language'])
    _novas_variaveis_dummy = _lb.classes_
    _df_variaveis_dummy = pd.DataFrame(_variaveis_dummy, columns=_novas_variaveis_dummy)
    previsores = previsores.join(_df_variaveis_dummy)
    previsores = previsores.drop('original_language',axis=1)
    
    # Variavel genre_main
    _variaveis_dummy = _lb.fit_transform(previsores['genre_main'])
    _novas_variaveis_dummy = _lb.classes_
    _df_variaveis_dummy = pd.DataFrame(_variaveis_dummy, columns=_novas_variaveis_dummy)
    previsores = previsores.join(_df_variaveis_dummy)
    previsores = previsores.drop('genre_main',axis=1)
    




# =============================================================================
#                 Separando em base de testes e treinamento
# =============================================================================

#  usando 25% para teste
from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = train_test_split(previsores, objetivo, test_size=0.25, random_state=69)





# # =============================================================================
# #                     Padronização dos dados
# # =============================================================================

if _PADRONIZACAO:
    from sklearn.preprocessing import StandardScaler
    
    _scaler = StandardScaler()
    previsores_treinamento = _scaler.fit_transform(previsores_treinamento)
    previsores_teste = _scaler.transform(previsores_teste)





