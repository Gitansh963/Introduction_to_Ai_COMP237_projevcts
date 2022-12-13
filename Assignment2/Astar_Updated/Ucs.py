'''
@author: Devangini Patel
'''
from State import State
from Node import Node
import queue

    

def performUcsSearch():
    """
    This method performs A* search
    """
    
    #create queue
    pqueue = queue.PriorityQueue()
    
    #create root node
    initialState = State()
    root = Node(initialState, None)
    
    
    #add to priority queue
    pqueue.put((root.costFromRoot, root))
    
    #check if there is something in priority queue to dequeue
    while not pqueue.empty(): 
        
        #dequeue nodes from the priority Queue
        _, currentNode = pqueue.get()
        
        #remove from the fringe
        currentNode.fringe = False
        
        #check if it has goal State
        print ("-- dequeue --", currentNode.state.place)
        
        #check if this is goal state
        if currentNode.state.checkGoalState():
            print ("reached goal state")
            #print the path
            print ("----------------------")
            print ("Path")
            currentNode.printPath()
            break
            
        #get the child nodes 
        childStates = currentNode.state.successorFunction()
        for childState in childStates:
            
            childNode = Node(State(childState), currentNode)
            
            #add to tree and queue
            pqueue.put((childNode.costFromRoot, childNode))
        
                
    #print tree
    print ("----------------------")
    print ("Tree")
    root.printTree()
    
performUcsSearch()