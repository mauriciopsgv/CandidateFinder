class TreeManipulator():
    def __init__(self, decision_tree, feature_list):
        self.estimator_ = decision_tree
        self.n_nodes = decision_tree.tree_.node_count
        self.children_left = decision_tree.tree_.children_left
        self.children_right = decision_tree.tree_.children_right
        self.feature = decision_tree.tree_.feature
        self.threshold = decision_tree.tree_.threshold
        self.feature_list = feature_list
        self.sample_values = [None] * len(feature_list)
        self.current_node = 0

    def get_feature_index(self):
        return self.feature[self.current_node]

    def get_feature_name(self):
        return self.feature_list[self.feature[self.current_node]]

    def am_i_on_a_leave(self):
        return self.children_left[self.current_node] == self.children_right[self.current_node]

    def go_to_left_child_(self):
        # This means feature <= threshold
        self.current_node = self.children_left[self.current_node]

    def go_to_right_child_(self):
        # This means feature > threshold
        self.current_node = self.children_right[self.current_node]

    def go_forward(self, new_value):
        self.sample_values[self.feature[self.current_node]] = new_value
        if new_value <= self.threshold:
            self.go_to_left_child_()
        else:
            self.go_to_right_child_()
