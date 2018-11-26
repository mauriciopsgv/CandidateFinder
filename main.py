# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 23:35:42 2018
"""

import pandas as pd
import time

from search_engine import SearchEngine

# Importing 

data = pd.read_csv('treated_dataset.csv')
data = data.drop('Unnamed: 0', axis=1)      # TODO: discover where this Unnamed comes from
data = data.sample(500, random_state=42)
# set up

early_adopted_features = ['professional_goals_salary_min',
       'professional_goals_salary_max', 
       #'professional_goals_contract_type', 'professional_goals_working_hours', 
       #'complementary_data_travel', 'complementary_data_residence', 
       'driving_A', 'driving_B', 'driving_C',
       'driving_D', 'driving_E', 'vehicle_Carro particular', 'vehicle_Moto',
       'vehicle_Caminhão', 'vehicle_Outro', 'language_Afrikaans',
       'language_Alemão', 'language_Bengalí', 'language_Cantonês',
       'language_Catalão', 'language_Chinês', 'language_Coreano',
       'language_Croato', 'language_Dinamarquês', 'language_Eslovaco',
       'language_Espanhol', 'language_Farsi', 'language_Finlandês',
       'language_Francês', 'language_Galego', 'language_Grego',
       'language_Hebraico', 'language_Holandês', 'language_Húngaro',
       'language_Indonésio', 'language_Inglês', 'language_Islandês',
       'language_Italiano', 'language_Japonês', 'language_Latin',
       'language_Latviano', 'language_Libras', 'language_Mandarin',
       'language_Noruego', 'language_Polonês', 'language_Português',
       'language_Punjabi', 'language_Romano', 'language_Russo',
       'language_Sueco', 'language_Swahili', 'language_Sânscrito',
       'language_Tagalog', 'language_Taiwanês', 'language_Tcheco',
       'language_Thai', 'language_Turco', 'language_Ucraniano',
       'language_Urdu', 'language_Vasco', 'language_Vietnamita',
       'language_Árabe', 'language_Índio']

early_adopted_features_options = {
    'professional_goals_salary_min': [],
    'professional_goals_salary_max': [],
    'driving_A': ['Nao', 'Sim'],
    'driving_B': ['Nao', 'Sim'],
    #'professional_goals_contract_type',
    #'professional_goals_working_hours', 
    #'complementary_data_travel',
    #'complementary_data_residence',
    'driving_C': ['Nao', 'Sim'],
    'driving_D': ['Nao', 'Sim'],
    'driving_E': ['Nao', 'Sim'],
    'vehicle_Carro particular': ['Nao', 'Sim'], 
    'vehicle_Moto': ['Nao', 'Sim'],
    'vehicle_Caminhão': ['Nao', 'Sim'], 
    'vehicle_Outro': ['Nao', 'Sim'], 
    'language_Afrikaans': ['Nao', 'Sim'], 
    'language_Alemão': ['Nao', 'Sim'], 
    'language_Bengalí': ['Nao', 'Sim'], 
    'language_Cantonês': ['Nao', 'Sim'],
    'language_Catalão': ['Nao', 'Sim'], 
    'language_Chinês': ['Nao', 'Sim'], 
    'language_Coreano': ['Nao', 'Sim'],
    'language_Croato': ['Nao', 'Sim'], 
    'language_Dinamarquês': ['Nao', 'Sim'], 
    'language_Eslovaco': ['Nao', 'Sim'],
    'language_Espanhol': ['Nao', 'Sim'], 
    'language_Farsi': ['Nao', 'Sim'], 
    'language_Finlandês': ['Nao', 'Sim'],
    'language_Francês': ['Nao', 'Sim'], 
    'language_Galego': ['Nao', 'Sim'], 
    'language_Grego': ['Nao', 'Sim'],
    'language_Hebraico': ['Nao', 'Sim'], 
    'language_Holandês': ['Nao', 'Sim'], 
    'language_Húngaro': ['Nao', 'Sim'],
    'language_Indonésio': ['Nao', 'Sim'], 
    'language_Inglês': ['Nao', 'Sim'], 
    'language_Islandês': ['Nao', 'Sim'],
    'language_Italiano': ['Nao', 'Sim'], 
    'language_Japonês': ['Nao', 'Sim'], 
    'language_Latin': ['Nao', 'Sim'],
    'language_Latviano': ['Nao', 'Sim'], 
    'language_Libras': ['Nao', 'Sim'], 
    'language_Mandarin': ['Nao', 'Sim'],
    'language_Noruego': ['Nao', 'Sim'], 
    'language_Polonês': ['Nao', 'Sim'], 
    'language_Português': ['Nao', 'Sim'],
    'language_Punjabi': ['Nao', 'Sim'], 
    'language_Romano': ['Nao', 'Sim'], 
    'language_Russo': ['Nao', 'Sim'],
    'language_Sueco': ['Nao', 'Sim'], 
    'language_Swahili': ['Nao', 'Sim'], 
    'language_Sânscrito': ['Nao', 'Sim'],
    'language_Tagalog': ['Nao', 'Sim'], 
    'language_Taiwanês': ['Nao', 'Sim'], 
    'language_Tcheco': ['Nao', 'Sim'],
    'language_Thai': ['Nao', 'Sim'], 
    'language_Turco': ['Nao', 'Sim'], 
    'language_Ucraniano': ['Nao', 'Sim'],
    'language_Urdu': ['Nao', 'Sim'], 
    'language_Vasco': ['Nao', 'Sim'], 
    'language_Vietnamita': ['Nao', 'Sim'],
    'language_Árabe': ['Nao', 'Sim'], 
    'language_Índio': ['Nao', 'Sim']
    }

y = data['id']
X = data[early_adopted_features]

engine = SearchEngine(early_adopted_features, early_adopted_features_options)
print("Starting to fit tree with all samples")
start = time.time()
engine.fit(X, y)
end = time.time()
print("Finished fitting in " + str(end - start)  + " seconds")
candidates_ids = engine.search()

print("Foram encontrados " + str(len(candidates_ids)) + " candidatos")

#print("Os candidatos que mais se encaixam na sua procura são:")
#print(data.columns)
#for i in candidates_ids:
#    print(data[data['id'] == i].values)

