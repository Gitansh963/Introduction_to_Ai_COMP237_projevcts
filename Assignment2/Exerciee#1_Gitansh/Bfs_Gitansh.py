# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 15:29:16 2022

@author: gitansh mittal
"""

from GraphData import graph
def bfs_gitansh(graph, start_node, goal_node):
  queue = []    
  visited = []
  trail= []
  visited.append(start_node)
  queue.append(start_node)

  while queue:          
    g = queue.pop(0) 
    trail.append(g)
    if g == goal_node:
        break

    for neighbour in graph[g]:
         if neighbour not in visited:
             visited.append(neighbour)
             queue.append(neighbour)
  if goal_node not in graph or start_node not in graph:
      print("\n Node not in graph")
  elif(goal_node not in trail):
        print("\n Goal Node is not in trail")
  elif start_node == goal_node:
      print("\n you are already there")
  else:
       for node in trail:
           print(node, end = " ")
        
bfs_gitansh(graph, 'Dolly','Gitansh')
print("\n second run")
bfs_gitansh(graph, 'George','Ema')
