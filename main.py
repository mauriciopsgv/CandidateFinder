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
       'driving_A', 'driving_B']
early_adopted_features_options = {
        'professional_goals_salary_min': [],
        'professional_goals_salary_max': [],
        'driving_A': ['Nao', 'Sim'],
        'driving_B': ['Nao', 'Sim']
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

