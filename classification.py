import numpy
import math


# These are suggested helper functions
# You can structure your code differently, but if you have
# trouble getting started, this might be a good starting point

# Create the decision tree recursively
def make_node(previous_ys, xs, ys, columns):
    # WARNING: lists are passed by reference in python
    # If you are planning to remove items, it's better 
    # to create a copy first
    columns = columns[:]

    # First, check the three termination criteria:
    
    # If there are no rows (xs and ys are empty): 
    #      Return a node that classifies as the majority class of the parent
    if len(xs) == 0 and len(ys) == 0:
        return {"type" : "class", "class" : majority(previous_ys)}

    # If all ys are the same:
    #      Return a node that classifies as that class 
    if same(ys):
        return {"type" : "class", "class" : ys[0]}

    # If there are no more columns left:
    #      Return a node that classifies as the majority class of the ys
    if len(columns) == 0:
        return {"type" : "class", "class" : majority(ys)}

    # Otherwise:
    # Compute the entropy of the current ys 
    # For each column:
    #     Perform a split on the values in that column 
    #     Calculate the entropy of each of the pieces
    #     Compute the overall entropy as the weighted sum 
    #     The gain of the column is the difference of the entropy before
    #        the split, and this new overall entropy 
    # Select the column with the highest gain, then:
    # Split the data along the column values and recursively call 
    #    make_node for each piece 
    # Create a split-node that splits on this column, and has the result 
    #    of the recursive calls as children.
    current_entro = entropy(ys)
    column_entro = {}
    for column in columns:
        pieces = {}
        index = 0
        for item in xs:
            if item[column] not in pieces.keys():
                pieces[item[column]] = [ys[index]]
            else:
                pieces[item[column]].append(ys[index])
            
            index += 1
        
        summation = 0
        for key in pieces:
            prob = len(pieces[key]) / len(xs)
            summation += prob * entropy(pieces[key])
        
        column_entro[column] = summation

    chosen_col = -1
    highest_gain = 0
    for key in column_entro:
        if chosen_col == -1 or column_entro[key] - current_entro < highest_gain:
            highest_gain = column_entro[key] - current_entro
            chosen_col = key

    columns.remove(chosen_col)

    branches = {}
    index = 0
    for item in xs:
        if item[chosen_col] not in branches.keys():
            branches[item[chosen_col]] = {"ys" : [ys[index]], "xs" : [xs[index].copy()]}
        else:
            branches[item[chosen_col]]["ys"].append(ys[index])
            branches[item[chosen_col]]["xs"].append(xs[index].copy())

        index += 1

    tree = {"type" : "split", "split" : chosen_col, "children" : {}}
    for key in branches:
        tree["children"].update( { key : make_node(ys, branches[key]['xs'], branches[key]['ys'], columns)} )

    return tree
    # Note: This is a placeholder return value
    #return {"type": "class", "class": majority(ys)}

    
    

# Determine if all values in a list are the same 
# Useful for the second basecase above
def same(values):
    if not values: 
        return True
    else:
        item_set = set(values)
        if len(item_set) == 1:
            return True
    
    return False

    # if there are values:
    # pick the first, check if all other are the same 
    
# Determine how often each value shows up 
# in a list; this is useful for the entropy
# but also to determine which values is the 
# most common
def counts(values):
    total = {}
    unique = set(values)
    
    for item in unique:
        total[item] = values.count(item)

    # placeholder return value 
    return total
   

# Return the most common value from a list 
# Useful for base cases 1 and 3 above.
def majority(values):
    total = counts(values)
    majority_value = -1
    current_total = -1

    for key in total:
        if(total[key] > current_total):
            current_total = total[key]
            majority_value = key

    # placeholder return value
    return majority_value
    
    
# Calculate the entropy of a set of values 
# First count how often each value shows up 
# When you divide this value by the total number 
# of elements, you get the probability for that element 
# The entropy is the negation of the sum of p*log2(p) 
# for all these probabilities.
def entropy(values):
    total = counts(values)
    total_amt = len(values)
    summation = 0

    for key in total:
        prob = total[key] / total_amt
        summation += (prob * numpy.log2(prob))

    summation *= -1

    # placeholder return value
    return summation

# This is the main decision tree class 
# DO NOT CHANGE THE FOLLOWING LINE
class DecisionTree:
# DO NOT CHANGE THE PRECEDING LINE
    def __init__(self, tree={}):
        self.tree = tree
    
    # DO NOT CHANGE THE FOLLOWING LINE    
    def fit(self, x, y):
    # DO NOT CHANGE THE PRECEDING LINE

        self.majority = majority(y)
        self.tree = make_node(y, x, y, list(range(len(x[0]))))
        
    # DO NOT CHANGE THE FOLLOWING LINE    
    def predict(self, x):
    # DO NOT CHANGE THE PRECEDING LINE    
        if not self.tree:
            return None

        predictions = []
        # To classify using the tree:
        # Start with the root as the "current" node
        # As long as the current node is an interior node (type == "split"):
        #    get the value of the attribute the split is performed on 
        #    select the child corresponding to that value as the new current node 

        # NOTE: In some cases, your tree may not have a child for a particular value 
        #       In that case, return the majority value (self.majority) from the training set 
        
        # IMPORTANT: You have to perform this classification *for each* element in x 
        
        for item in x:
            current = self.tree
            while current["type"] == "split":
                split = current["split"]
                choice = item[split]
                if choice in current["children"]:
                    current = current["children"][choice]
                else:
                    predictions.append(self.majority)
                    break
            if(current["type"] != "split"):
                predictions.append(current["class"])

        # placeholder return value
        # Note that the result is a list of predictions, one for each x-value
        return predictions
    
    # DO NOT CHANGE THE FOLLOWING LINE
    def to_dict(self):
    # DO NOT CHANGE THE PRECEDING LINE
        # change this if you store the tree in a different format
        return self.tree
       