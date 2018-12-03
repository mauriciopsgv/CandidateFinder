# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:48:34 2018

@author: mauri
"""
import numpy as np

from interface import Interface
from tree_manipulator import TreeEngine
from gensim.models import KeyedVectors


class SearchEngine():
    def __init__(self, features_name, feature_options):
        self.tree_engine = TreeEngine(features_name)
        self.interface = Interface(feature_options)
        self.word_model = KeyedVectors.load_word2vec_format('cbow_s50.txt')
        self.samples = None
        self.titles = None
        self.target = None
        self.titles_unique = None
        self.title_vectors = None
        
    def is_clean(self, word):
        return word != '/' and len(word) > 2 
        
    def clean_word(self, word):
        bad_characters = [ '(', ')', ',', '.', '"', "'", '-', '\r', '\n', '*', ';', 'ð', '\t']
        for character in bad_characters:
            if character in word:
                word = word.replace(character, '')
        if word == 'aeropespacial':
            word = 'aeroespacial'
        elif word == 'serralheiria':
            word = 'serralheira'
        elif word == 'mandrilagem':
            word = 'mecânico'
        elif word == 'colorimetrista':
            word = 'colorista'
        elif word == 'fresaria':
            word = 'fresa'
        elif word == 'traumatorpedia':
            word = 'trauma'
        elif word == 'neorologia':
            word = 'neurologia'
        elif word == 'censoriamento':
            word = 'sensoriamento'
        return word

    def clean_title(self, word_list):
        return [self.clean_word(word) for word in word_list]
    
    def get_phrase_vector(self, phrase):
        acumulator = np.zeros(50)
        for word in phrase:
            acumulator = acumulator + self.word_model.word_vec(word)
        phrase_vector = np.divide(acumulator, len(phrase)) if len(phrase) != 0 else acumulator
        return phrase_vector if np.linalg.norm(phrase_vector) == 0 else np.divide(phrase_vector, np.linalg.norm(phrase_vector))
    
    def update_title_vectors(self):
        self.titles_unique = self.titles.unique()
        splitted_list = [ title.split(' ')[1:] for title in self.titles_unique ]
        final_list = [ [ i.lower() for i in title if self.is_clean(i)] 
                        if ( '/' in title or len(title) > 1 ) else [title[0].lower()] for title in splitted_list ]
        clean_titles = [self.clean_title(title) for title in final_list]
        self.title_vectors = [self.get_phrase_vector(phrase) for phrase in clean_titles]

    def fit(self, X, y):
        self.titles = X['title']
        self.update_title_vectors()
        self.samples = X.drop('title', axis=1)
        self.target = y
        self.tree_engine.fit(self.samples, self.target)
        
    def get_close_titles(self, word_to_find):
        vector = self.get_phrase_vector(word_to_find)
        distances = [np.absolute(np.linalg.norm(vector - candidate_vector)) 
                              for candidate_vector in self.title_vectors]
        return distances
        
    def filter_dataset_for_position(self, position):
        word_to_find = [position]   # need to split by space later on        
        distances = self.get_close_titles(word_to_find)
        distances_arg_sorted = np.argsort(distances)
        for i in range(0,10):
            #display(clean_titles[distances_arg_sorted[i]])
            print(distances[distances_arg_sorted[i]])

    def search(self): # TODO: be sure tree is fitted before searching
        position = input("Input a position: ")
        self.filter_dataset_for_position(position)
        print("Voce escolheu a posicao " + position)
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

        