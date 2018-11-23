# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:48:34 2018

@author: mauri
"""
import numpy as np

from interface import Interface
from tree_manipulator import TreeEngine

class SearchEngine():
    def __init__(self, features_name, feature_options):
        self.tree_engine = TreeEngine(features_name)
        self.interface = Interface(feature_options)
        self.samples = None
        self.target = None
        
    def fit(self, X, y):
        self.samples = X
        self.target = y
        self.tree_engine.fit(X,y)
    
    def search(self): # TODO: be sure tree is fitted before searching
        test = input("Input a position: ")
        print("Voce escolheu a posicao " + test)
        self.tree_engine.reset_path()
        while (not self.tree_engine.am_i_on_a_leave()):
            feature_name = self.tree_engine.get_feature_name()
            self.interface.ask_question(feature_name)
            if (not self.tree_engine.already_have_feature()):
                self.interface.print_feature_options(feature_name)
                option_index = float(input())
                print("Voce optou pela opcao " + str(option_index))
                if option_index == -1:
                    self.samples = self.samples.drop(feature_name, axis=1)
                    self.tree_engine.refit(self.samples, self.target)
                    self.tree_engine.reset_path()
                else:
                    self.tree_engine.go_forward(option_index)
            else:
                self.tree_engine.go_forward()
        
        print('Node id que caiu foi = ' + str(self.tree_engine.current_node))
        return self.tree_engine.get_candidates_id()
