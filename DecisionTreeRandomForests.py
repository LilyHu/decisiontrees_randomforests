import numpy as np

'''
Basic binary node
'''
class Node(object):
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.split_rule =[None, None] # this is a list. python doesn't have fixed type arrays!
        self.label = None

'''
Standalone Decision Tree
'''        
class DecisionTree(object):
    def __init__(self, max_tree_depth):
        self.root = None
        self.max_tree_depth = max_tree_depth
    
    def train(self, data, labels):
        self.root = growTree(data, labels, self.max_tree_depth)
            
    def predict(self, test_data):
        num_test_data = len(test_data)
        predictions = []
        path = []
        for sample_i in xrange(0, num_test_data):
            current_node = self.root
            while current_node.label is None:
                feature_to_split_on = current_node.split_rule[0]
                feature_threshold = current_node.split_rule[1]
                if test_data[sample_i, feature_to_split_on] < feature_threshold:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child
            predictions.append(current_node.label)        
        return np.asarray(predictions)

    def tracePredict(self, test_data):
        #num_test_data = len(test_data)
        predictions = []
        path = []
        
        current_node = self.root
        while current_node.label is None:
            feature_to_split_on = current_node.split_rule[0]
            feature_threshold = current_node.split_rule[1]
            if test_data[feature_to_split_on] < feature_threshold:
                current_node = current_node.left_child
                path.append([current_node.split_rule, '<'])
            else:
                current_node = current_node.right_child
                path.append([current_node.split_rule, '>'])
        prediction = current_node.label        
        return prediction, path

'''
Ramdom Forest
'''
class RandomForest(object):
    def __init__(self, num_trees, max_tree_depth, num_samples_per_tree, num_possible_features_per_split):
        self.trees = []
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.num_samples_per_tree = num_samples_per_tree
        self.num_possible_features_per_split = num_possible_features_per_split
        
    def train(self, data, labels):
        for tree_i in xrange(0, self.num_trees):
            r_data, r_labels = randomizeDataMatrix(data, labels, self.num_samples_per_tree, True)
            new_tree = RandomizedDecisionTree(self.max_tree_depth, self.num_possible_features_per_split)
            new_tree.train(r_data, r_labels)
            self.trees.append(new_tree)
        
    def predict(self, test_data):
        votes = np.zeros((len(test_data), self.num_trees))
        predictions = []
        for tree_i in xrange(0, self.num_trees):
            votes[:,tree_i] = self.trees[tree_i].predict(test_data)
        for sample_i in xrange(0, len(test_data)):
            prediction = getMode(votes[sample_i, :])
            predictions.append(prediction)
        return np.asarray(predictions)
    
'''
RandomizedDecisionTree extends DecisionTree with a new self.root grown using growRandomizedTree
instead of growTree
'''
class RandomizedDecisionTree(DecisionTree):
    def __init__(self, max_tree_depth, num_possible_features_per_split):
        self.root = None
        self.max_tree_depth = max_tree_depth
        self.num_possible_features_per_split = num_possible_features_per_split
    
    def train(self, data, labels):
        self.root = growRandomizedTree(data, labels, self.max_tree_depth, self.num_possible_features_per_split)    

'''
MOdieifed from growTree to call randomizedSegmentor instead of segmentor
'''        
def growRandomizedTree(data, labels, max_tree_depth, num_possible_features_per_split):
    node = Node()
    split_rule, node_label = randomizedSegmentor(data, labels, num_possible_features_per_split)
    
    if max_tree_depth <= 0: # stop recursion
        node.label = getMode(labels)
    
    elif split_rule == None:
        node.label=node_label
        # This node is a leaf node
    else:
        
        data_left_child, labels_left_child, data_right_child, labels_right_child = divideDataForChildren(data, labels, split_rule)
        if (len(data_left_child)==0) or ((len(data_right_child)==0)):
            node.label=getMode(labels)
        else:
            node.split_rule = split_rule
            node.right_child = growRandomizedTree(data_right_child, labels_right_child, max_tree_depth-1, num_possible_features_per_split)
            node.left_child = growRandomizedTree(data_left_child, labels_left_child, max_tree_depth-1, num_possible_features_per_split)
        
    return node
        
'''
Recurisvely adds nodes
'''               
def growTree(data, labels, max_tree_depth):
    node = Node()
    split_rule, node_label = segmentor(data, labels)
    
    if max_tree_depth <= 0: # stop recursion
        node.label = getMode(labels)
    
    elif split_rule == None:
        node.label=node_label
        # This node is a leaf node
    else:
                
        data_left_child, labels_left_child, data_right_child, labels_right_child = divideDataForChildren(data, labels, split_rule)
        if (len(data_left_child)==0) or ((len(data_right_child)==0)):
            node.label=getMode(labels)
        else:    
            node.split_rule = split_rule
            data_left_child, labels_left_child, data_right_child, labels_right_child = divideDataForChildren(data, labels, split_rule)
            node.right_child = growTree(data_right_child, labels_right_child, max_tree_depth-1)
            node.left_child = growTree(data_left_child, labels_left_child, max_tree_depth-1)

    return node

'''
Assign labels to a node using the mode of the labels
'''
def getMode(labels):
    label_dict = {}
    num_labels = len(labels)
    for sample_i in xrange(0,num_labels):
        if labels[sample_i] not in label_dict:
            label_dict[labels[sample_i]] = 1
        else:
            label_dict[labels[sample_i]] += 1
    
    max_label = None
    max_count = 0
    for key_i in label_dict:
        if label_dict[key_i] > max_count:
            max_count = label_dict[key_i]
            max_label = key_i
    return max_label
    
    
'''
Creates data sets for the left and right children of a parent node after a split is determined
Output: 
    data_left_child
    labels_left_child
    data_right_child
    labels_right_child
    
Extension Notes
    Binary split
    Split on single feature
'''
def divideDataForChildren(data, labels, split_rule):
    num_data = len(data)
    split_rule_feature = split_rule[0]
    split_rule_threshold = split_rule[1]
    data_left_child = []
    labels_left_child = []
    data_right_child = []
    labels_right_child = []
    for sample_i in xrange(0, num_data):
        if data[sample_i, split_rule_feature] < split_rule_threshold:
            data_left_child.append(data[sample_i,:])
            labels_left_child.append(labels[sample_i])
        else:
            data_right_child.append(data[sample_i, :])
            labels_right_child.append(labels[sample_i])
    return np.asarray(data_left_child), np.asarray(labels_left_child), np.asarray(data_right_child), np.asarray(labels_right_child)
    
'''
Calcualtes the impurity due to a split
Remember that the lower the impurity, the better
Input:  
    left_label_hist    2 element array-like
    right_label_hist   2 element array-like

Extension Notes:
    Binary splits only
        
'''    
def impurity(left_label_hist, right_label_hist):
    total_children = float(sum(left_label_hist)+sum(right_label_hist))
    impurity = sum(left_label_hist)/total_children*entropyOfNode(left_label_hist)+sum(right_label_hist)/total_children*entropyOfNode(right_label_hist)
    return impurity

'''
Calculates the entropy of a node
Input: 
    hist      an array-like object of size 2
    
Extension Notes:
    Binary splits only
'''
def entropyOfNode(hist):
    if (hist[0] == 0) or (hist[1] == 0):
        entropy = 0
    else: 
        total_children = float(sum(hist))
        entropy = -(hist[0]/(total_children)*math.log(hist[0]/(total_children),2)+hist[1]/(total_children)*math.log(hist[1]/(total_children),2))
    return entropy

'''
Returns best split and node_label, one of which will be None
If all the data is sorted, returns the node_label

Extension notes
    What if split_rule = None and node_labe = None because all splits give impurity = 1???
'''
def segmentor(data, labels):
    
    num_features = shape(data)[1]
    min_impurity = 1
    split_rule = None
    node_label = None
    for feature_i in xrange(0,num_features):
        node_split_feature_threshold, node_label_on_feature = splitSingleFeature(data[:,feature_i], labels)
        if node_split_feature_threshold is None: # Check if all the elements are classified
            split_rule = None
            node_label = node_label_on_feature
            return (split_rule, node_label)
        else:
            left_label_hist, right_label_hist = createLabelHist(data[:, feature_i], node_split_feature_threshold, labels)
            #return left_label_hist, right_label_hist
            impurity_of_feature = impurity(left_label_hist, right_label_hist)
            # Check if this feature minimizes the impurity 
            if impurity_of_feature < min_impurity:
                split_rule = [feature_i, node_split_feature_threshold]
                node_label = None
                min_impurity = impurity_of_feature
    if (split_rule==None) and (node_label==None):
        node_label = getMode(labels)
    return (split_rule, node_label)

'''
Modified from segmentor to pick a split from a subset of features
'''
def randomizedSegmentor(data, labels, num_possible_features_per_split):
    #print shape(data)
    num_features = shape(data)[1]
    min_impurity = 1
    split_rule = None
    node_label = None
    possible_features = np.random.choice(range(0,num_features), num_possible_features_per_split)
    
    for feature_i in possible_features:
        node_split_feature_threshold, node_label_on_feature = splitSingleFeature(data[:,feature_i], labels)
        if node_split_feature_threshold is None: # Check if all the elements are classified
            split_rule = None
            node_label = node_label_on_feature
            return (split_rule, node_label)
        else:
            left_label_hist, right_label_hist = createLabelHist(data[:, feature_i], node_split_feature_threshold, labels)
            #return left_label_hist, right_label_hist
            impurity_of_feature = impurity(left_label_hist, right_label_hist)
            # Check if this feature minimizes the impurity 
            if impurity_of_feature < min_impurity:
                split_rule = [feature_i, node_split_feature_threshold]
                node_label = None
                min_impurity = impurity_of_feature
    if (split_rule==None) and (node_label==None):
        node_label = getMode(labels)
    return (split_rule, node_label)

'''
Calculates the split threshold on a single feature
Input: 
    data is a vector of one feature
'''    
def splitSingleFeature(data, labels):
    '''
    IMPROVEMENT: A better way to do this is to arrange the points from smallest to largest and then
    test each split as where the labels change
    '''
    # Split the data by label into a dict with labels as the keys
    data_dict = {}
    n = len(data)
    for i in xrange(0, n):
        if labels[i] in data_dict:
            data_dict[labels[i]].append(data[i])
        else:
            data_dict[labels[i]]=[data[i]]
        '''
        if labels[i] not in data_dict:
            data_dict[labels[i]] = []
        data_dict[labels[i]].append(data[i])
        '''
    # Calculate the mean of the labels
    mean_labels = {}
    for label in data_dict: # this is python magic! It loops through the keys without writting data_dict.keys()
        mean_labels[label] = np.mean(data_dict[label], axis=0)
    # Check if a leaf node
    if len(data_dict) == 1:
        node_label = data_dict.keys()[0] # This is a leaf node
        node_split_feature_threshold = None
    else: 
        # Pick the threshold as the mean of means
        # assums binary class split
        node_split_feature_threshold = mean(mean_labels.values())
        node_label = None
        
    return (node_split_feature_threshold, node_label)


'''
Creates a histogram of the data by label
Input: 
    data                     vector
    node_feature_threshold   threshold to split on
    labels                   labels corresponding to the vector
    
Output: 
    left_label_dict.values()   histogram of the frequencies
    right_label_dict.values()  histogram of the frequencies in the right child

Extension notes:
    Because the histograms are first implemented as dictionaries,
    this function is ready to take on multiway splits

'''
def createLabelHist(data, node_split_feature_threshold, labels):
    num_data = len(data)
    left_label_dict = {}
    right_label_dict = {}
    
    for data_i in xrange(0, num_data):
        # Make sure the left_label_hist and right_label_hist have the same keys
        if labels[data_i] not in left_label_dict:
            left_label_dict[labels[data_i]] = 0
            right_label_dict[labels[data_i]] = 0
        # Enter the new data point into its corresponding histogram
        if data[data_i] < node_split_feature_threshold:
            left_label_dict[labels[data_i]] += 1
        else:
            right_label_dict[labels[data_i]] +=1
    return left_label_dict.values(), right_label_dict.values()

    