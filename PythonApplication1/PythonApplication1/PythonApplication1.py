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
queue = deque() # nwe queue datastructuur mk
# bedoeling om met bfs een opsomming eerst te maken. 

def printer(node):
    huidige_node = node
    try: # checks whether any child nodes exist on the left hand side
        volgende_node_l = huidige_node.left
        som = (np.sum(volgende_node_l.y))
        lengte = (len(volgende_node_l.y))
        # class distribution within left child node
        print("going left: (",lengte-som,", ", som,")")

        if volgende_node_l not in queue:
            queue.append(volgende_node_l)

    
    #except:
      #  pass

   # try: # checks whether any child nodes exist on the right hand side
        volgende_node_r = huidige_node.right
        som = (np.sum(volgende_node_r.y))
        lengte = (len(volgende_node_r.y))
        # class distribution within right child node
        print("going right: (",lengte-som,", ", som,")")

        if volgende_node_r not in queue:
            queue.append(volgende_node_r)
            
    except:
        pass
        
root = (valuetree)
queue.append(root)

#print("root:  ", root.y)
som = (np.sum(root.y))
lengte = (len(root.y))
print("starting from: (",lengte-som,", ", som,")")

while len(queue) != 0:
    printer(node=queue.pop())
