import numpy as np

#TreeNode class used in the tree_grow function
class TreeNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.left = None
        self.right = None

#Impurity function for binary class labels
def impurity(array):
    a = 1 - (np.sum(array[:] == 0) / len(array))**2 - (np.sum(array[:] == 1) / len(array))**2
    return a


#bestsplit: NOTE: x is now a 2d numpy array with attribute values
#returns a tuple with the best column to split on with its associated split value
#TO DO: implement nfeat paramater for random forests (now it just assumes nfeat is always equal to the number of columns in x)
def bestsplit(x,y, minleaf, nfeat):

    best_reduction = 0
    best_column = -1 #The featured column that has the best split
    best_value = -1 #The associated value to splt on

    if len(x.shape) == 1:
        size = 1
    else:
        size = x.shape[1]

    for column in range(x.shape[1]):#For loop that considers each of the columns.
        xcolumn = x[:, column]

        #Find splitpoints
        sorted_xcolumn = np.unique(xcolumn) #Sort and get rid of duplicate values
        splitpoints = (sorted_xcolumn[0:len(sorted_xcolumn)-1]+sorted_xcolumn[1:len(sorted_xcolumn)])/2

        #Calculate the best split
        for i in splitpoints:
            l = y[xcolumn <= i]
            r = y[xcolumn > i]

            #Skip the false split with less than minleaf elements in l or r
            if len(l) < minleaf or len(r) < minleaf:
                continue

            #Calculate the impurity reduction of the split and remember the best
            impurity_reduction = impurity(y) - (len(l)/len(y))*impurity(l) - (len(r)/len(y))*impurity(r)
            if impurity_reduction > best_reduction:
                best_reduction = impurity_reduction
                best_value = i
                best_column = column
    return (best_column, best_value)


#recursive_growth recursively grows a tree starting from a node
def recursive_growth(node, nmin, minleaf, nfeat):
    x = node.x
    y = node.y

    #Growth stops if the node is a leaf node
    if len(y) < nmin:
        return

    #Find the best split with its associated column and value
    (column, value) = bestsplit(x, y, minleaf, nfeat)

    #Growth stops if no possible split was found
    if column == -1:
        return

    #Take all values in x and y for which the values in the associated column of x are lower/higher than the value "value", and put them in the child nodes
    leftcondition = x[:, column] <= value
    rightcondition = x[:, column] > value
    node.left = TreeNode(x[leftcondition], y[leftcondition])
    node.right = TreeNode(x[rightcondition], y[rightcondition])

    #Continue growing in the child nodes
    recursive_growth(node.left, nmin, minleaf, nfeat)
    recursive_growth(node.right, nmin, minleaf, nfeat)



def tree_grow(x, y, nmin, minleaf, nfeat):
    #nmin if a node contains fewer cases than nmin, it is a leaf node
    #minleaf: a leaf node must contain at least minleaf cases
    #nfeat: the number of features that should be considered for each split (Normally, this is equal to the number of columns of x), for random forest some random columns are selected for each split

    tree = TreeNode(x, y)
    recursive_growth(tree, nmin, minleaf, nfeat)

    return tree
 

#TO DO
def tree_pred(x, tr):
    return






#Example results from the credit scoring data seen in lecture 1. All nodes in the tree are printed. #TO DO: a nice print function for a tree would be nice

#Input credit scoring from lectures
'''
#valuex = np.array([[22, 28000],
#              [46, 32000],
#              [24, 24000],
#              [25, 27000],
#              [29, 32000],
#              [45, 30000],
#              [63, 58000],
#              [36, 52000],
#              [23, 40000],
#              [50, 28000]])
#valuey = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
'''
# These two lines do the same
valuex = np.array((credit_data[:, [0,1,2,3,4]]))
valuey = np.array((credit_data[:, [5][0]]))

valuenmin = 2
valueminleaf = 1
valuenfeat = 5 #number of columns in x

'''
# This code can be replaced
#Output credit scoring, tree is the same as the one in the lecture as should be according to the recommneded test given in the assignment
valuetree = tree_grow(valuex, valuey, valuenmin, valueminleaf, valuenfeat)
print("Root node")
print(valuetree.x)
print("Left split")
print(valuetree.left.x)
print("Right split")
print(valuetree.right.x)
print("Left left split")
print(valuetree.left.left.x)
print("Left right split")
print(valuetree.left.right.x)
print("Left right left split")
print(valuetree.left.right.left.x)
print("Left right right split")
print(valuetree.left.right.right.x)
'''

# Here follows a general code to this problem
from collections import deque
from treelib import Node, Tree

node_parent_dict = dict()
tree = Tree()   # t.b.v. de visualisatie

queue = deque() # nieuwe queue datastructuur maken

def represent_node_content(node):
    som = (np.sum(node.y))
    lengte = (len(node.y))
    return str((round((lengte-som)), round(som)))

def printer(node):
    huidige_node = node
    try: 
        # checks whether any child nodes exist on the left hand side
        volgende_node_l = huidige_node.left
        print(represent_node_content(volgende_node_l))
        queue.append(volgende_node_l)
        kv_vgd_l = {represent_node_content(volgende_node_l): represent_node_content (huidige_node)}
        node_parent_dict.update(kv_vgd_l)

        # checks whether any child nodes exist on the right hand side
        volgende_node_r = huidige_node.right
        print(represent_node_content(volgende_node_l))
        queue.append(volgende_node_r)
        kv_vgd_r = {represent_node_content(volgende_node_r): represent_node_content(huidige_node)}
        node_parent_dict.update(kv_vgd_r)
    except:
        pass

root = (valuetree)
queue.append(root)


print(represent_node_content(root))

while len(queue) != 0:
    printer(node=queue.pop())


# CAUTION: getting this code to work requires the following action
# INSTALL TREELIB==1.6.4 AND FOLLOW THE INSTRUCTIONS HERE:
# https://stackoverflow.com/questions/46345677/treelib-prints-garbage-instead-of-pseudographics-in-python3

tree.create_node((represent_node_content(root)),(represent_node_content(root)))

for key, value in node_parent_dict.items():
    tree.create_node(key, key , parent=value)
tree.show()
