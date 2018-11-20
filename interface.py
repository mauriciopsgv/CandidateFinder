# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:48:36 2018

@author: mauri
"""

class Interface():
    def __init__(self, question_options):
        self.question_options = question_options
    
    def print_feature_options(self, feature_name):
        options = self.question_options[feature_name]
        print(' -1 - Nao faz diferenca')
        for index, option in enumerate(options):
            print(str(index) + ' - ' + option)
            
    def ask_question(self, feature_name):
        print('Qual valor voce gostaria para ' + feature_name)

