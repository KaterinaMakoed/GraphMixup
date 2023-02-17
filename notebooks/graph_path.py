import networkx as nx
from torch_geometric.utils.convert import to_networkx,from_networkx

import matplotlib.pyplot as plt
import numpy as np
import torch
import ot
import copy
import seaborn as sns
from torch_geometric.data.data import Data
import random
import graph_lib as gr


def get_cost_matrix(n):

    m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            m[i][j] = abs(i-j)
    return m 

def update_path(path):
    for i in range(path.shape[0]):
        for j in range(path.shape[1]):
            diff = j - i
            if (diff >= 2) & (path[i][j] > 0):
                value = path[i][j]
                path[i][j] = 0
                j_ind = diff - 1
                i_ind = diff - 1 - j_ind

                for k in range(diff):
                    print(diff)
                    print(i+i_ind, j-j_ind)
                    path[i+i_ind][j-j_ind] += value 
                    j_ind -= 1
                    i_ind += 1
                    
            if (diff <= -2) & (path[i][j] > 0):
                value = path[i][j]
                path[i][j] = 0
                j_ind = diff - 1
                i_ind = diff - 1 - j_ind

                for k in range(diff):
                    print(diff)
                    print(i-i_ind, j+j_ind)
                    path[i-i_ind][j+j_ind] += value 
                    j_ind -= 1
                    i_ind += 1                   
                    

                print(i, j)
    return path

def select_action(prob_list, len_list, actions = [0, 1, 2]):
    print(prob_list, len_list)
    weights = [prob_list[i] * len_list[i] for i in range(len(prob_list))]
    print("weights: ", weights)
    action = random.choices(actions, weights=weights, k=1)
    # print(action)
    return action[0]

def select_nodes(path, nodes, actions = [0, 1, 2]):

    prob_list = [path[0][1], path[1][2], path[2][3] ]
    len_list = [len(node) for node in nodes]
    print(prob_list, len_list)
    action_1 = select_action(prob_list, len_list, actions = actions)
    
    prob_list[action_1] -= 1
    len_list[action_1] -= 1
    
    action_2 = select_action(prob_list, len_list, actions = actions)
    
    
    # action_1, action_2 = random.choices(actions, weights=weights, k=2)
    print('actions: ', action_1, action_2)
    is_same_node = True
    while is_same_node: 
        node_1 = random.choice(nodes[action_1])
        node_2 = random.choice(nodes[action_2])
        is_same_node = node_1 == node_2
    return action_1, action_2, node_1, node_2


def get_nodes(g):
    dict_degrees = dict(g.degree)
    nodes = []
    for i in range(max(dict_degrees.values())+1):
        node_i = list(filter(lambda x: dict_degrees[x] == i, dict_degrees))
        nodes.append(node_i)
    return nodes

def run_step(g1, path):
    # get nodes
    nodes = get_nodes(g1)
    print("all nodes: ", nodes)
    # select actions and nodes 
    action_1, action_2, node_1, node_2 = select_nodes(path, nodes, actions = [0, 1, 2])
    print("Nodes: ", node_1, node_2)
    # add edge
    if g1.get_edge_data(node_1, node_2, default=True): # edge doesnot exist 
        g1.add_edge(node_1, node_2)
        # update path
        path[action_1][action_1+1] -= 1
        path[action_2][action_2+1] -= 1
    path_sum = path[0][1] + path[1][2] + path[2][3]
    print("Path sum: ", path_sum)
    return g1, path, path_sum

def run_check(g, d1, d2):
    
    data = from_networkx(g)
    data.x = torch.ones(len(list(g.nodes)), 1)
    gr.draw_graph(data)
    
    print("Distance between original graphs: ", gr.TMD_original(d1, d2, L=2, w=1)[3])
    print("Distance to the first graph: ", gr.TMD_original(d1, data, L=2, w=1)[3])
    print("Distance to the second graph: ", gr.TMD_original(d2, data, L=2, w=1)[3])
    
    print('*' * 10)
    print("Distance between original graphs, level = 3: ", gr.TMD_original(d1, d2, L=3, w=1)[3])
    print("Distance to the first graph, level = 3: ", gr.TMD_original(d1, data, L=3, w=1)[3])
    print("Distance to the second graph, level = 3: ", gr.TMD_original(d2, data, L=3, w=1)[3])
    
    
    print('*' * 10)
    print("Distance between original graphs, level = 4: ", gr.TMD_original(d1, d2, L=4, w=1)[3])
    print("Distance to the first graph, level = 4: ", gr.TMD_original(d1, data, L=4, w=1)[3])
    print("Distance to the second graph, level = 4: ", gr.TMD_original(d2, data, L=4, w=1)[3])