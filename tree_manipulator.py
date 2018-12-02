import numpy as np
from sklearn.tree import DecisionTreeClassifier

class TreeEngine():
    def __init__(self, features_name):
        self.estimator_ = DecisionTreeClassifier(criterion='entropy', max_depth=6 ,random_state=42)
        self.features_name = features_name
        self.sample_values = [None] * len(features_name)
        self.current_node = 0
        self.classes = []
        
    def fit(self, X, y):
        self.classes = np.sort(y.values)
        self.estimator_.fit(X,y)
        self.update_tree_attributes()
        
    def refit(self, X, y):
        feature_index = self.get_feature_index()
        self.sample_values = self.sample_values[:feature_index] + self.sample_values[feature_index + 1:]
        self.features_name = self.features_name[:feature_index] + self.features_name[feature_index + 1:]
        self.fit(X,y)
        
    def update_tree_attributes(self):
        self.n_nodes = self.estimator_.tree_.node_count
        self.children_left = self.estimator_.tree_.children_left
        self.children_right = self.estimator_.tree_.children_right
        self.feature = self.estimator_.tree_.feature
        self.threshold = self.estimator_.tree_.threshold
        self.values = self.estimator_.tree_.value

    def reset_path(self):
        self.current_node = 0

    def get_feature_index(self):
        return self.feature[self.current_node]

    def get_feature_name(self):
        return self.features_name[self.feature[self.current_node]]
    
    def get_node_values(self):
        return self.values[self.current_node][0]
    
    def get_candidates_id(self):
        candidates_index = [item for sublist in np.argwhere(self.get_node_values()) for item in sublist] 
        return [self.classes[candidate_index] for candidate_index in candidates_index]

    def am_i_on_a_leave(self):
        return self.children_left[self.current_node] == self.children_right[self.current_node]

    def already_have_feature(self):
        return self.sample_values[self.feature[self.current_node]] != None

    def go_to_left_child_(self):
        # This means feature <= threshold
        self.current_node = self.children_left[self.current_node]

    def go_to_right_child_(self):
        # This means feature > threshold
        self.current_node = self.children_right[self.current_node]

    def go_forward(self, new_value=None):
        # TODO: Revisit this boolean logic, it seems that it can be better
        # idea: make this function keep going through the tree and only halts when doesnt
        # have the feature cached or get to a leave, it will simplify the interface code
        if new_value != None or self.already_have_feature():
            cached_feature_value = self.sample_values[self.feature[self.current_node]]
            self.sample_values[self.feature[self.current_node]] = cached_feature_value if cached_feature_value else new_value
        else:
            raise Exception("There is no value for the given feature to be able to move forward")        
        self.go_forward_()
        
    def go_forward_(self):
        if self.sample_values[self.feature[self.current_node]] <= self.threshold[self.current_node]:
            self.go_to_left_child_()
        else:
            self.go_to_right_child_()            
            
    def get_prediction(self):
        if not self.am_i_on_a_leave():
            return -1
        return self.estimator_.predict([self.sample_values])
