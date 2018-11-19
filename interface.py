# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 23:35:42 2018
"""

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from tree_manipulator import TreeEngine


def print_feature_options(options):
    print(' -1 - Nao faz diferenca')
    for index, option in enumerate(options):
        print(str(index) + ' - ' + option)

data = pd.read_csv('treated_dataset.csv')
data = data.drop('Unnamed: 0', axis=1)
data = data.sample(500, random_state=42)

decision_tree = DecisionTreeClassifier()
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
decision_tree = decision_tree.fit(X, y)

handler = TreeEngine(decision_tree, early_adopted_features)

test = input("Input a position: ")
print("Voce escolheu a posicao " + test)
handler.reset_path()
while (not handler.am_i_on_a_leave()):
    print(handler.generate_question())
    feature = handler.get_feature_name()
    print_feature_options(early_adopted_features_options[feature])
    option_index = float(input())
    print("Voce optou pela opcao " + str(option_index))
    
    # Hard set stuff that will need change later
    handler.go_forward(option_index)
print('Id previsto foi ' + str(handler.get_prediction()))
    