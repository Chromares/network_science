import graph_tool as gt
import graph_tool.centrality
import graph_tool.clustering
import csv
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from heapq import nlargest
import pickle
from collections import defaultdict

grap = gt.Graph()
df = pd.read_csv('name_name.csv', dtype={"reps": np.int64, "comm" : str, "auth" : str})

#Data Cleaning
df.isnull().values.sum()

nan_rows = df[df.isnull().any(1)]
nan_rows

df = df.dropna()
df.isnull().values.sum()

#NXGraph
dfn = df.sample(10000)
del df
graph = nx.from_pandas_edgelist(dfn, source = 'comm', target = 'auth', edge_attr = 'reps',create_using = nx.DiGraph())
del dfn

def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, unicode):
        # Encode the key as ASCII
        key = key.encode('ascii', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, unicode):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG

def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, unicode):
        tname = 'string'

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key

gtG = nx2gt(graph)

gtG.list_properties()
pm = gtG.ep.properties['e', 'reps']

pgd, pg_itr = gt.centrality.pagerank(gtG, damping=0.85, pers=None, weight=pm, prop=None, epsilon=1e-06, max_iter=100, ret_iter=True)

ver_bet, edge_bet = gt.centrality.betweenness(gtG, pivots=None, vprop=None, eprop=None, weight=pm, norm=True)
ver_arr_bet = list(ver_bet.get_array())
with open('bet.pkl', 'wb') as f:
    pickle.dump(ver_arr_bet, f)

ver_close = gt.centrality.closeness(gtG, weight=pm, source=None, vprop=None, norm=True, harmonic=False)
ver_arr_close = list(ver_close.get_array())
with open('close.pkl', 'wb') as f:
    pickle.dump(ver_arr_close, f)


eigen, auth, hub = gt.centrality.hits(gtG, weight=pm, xprop=None, yprop=None, epsilon=1e-06, max_iter=100)

vertex_freq_x, vertex_freq_y = gt.stats.vertex_hist(gtG, "total", bins=[0, 1], float_count=True)

local_clustercoeff = gt.clustering.local_clustering(gtG, prop=None, undirected=True)

global_coeff = gt.clustering.global_clustering(gtG)


#Distributions
counter_bet = defaultdict(int)
counter_close = defaultdict(int)

for i in range(0,len(ver_arr_bet)):
    counter_bet[ver_arr_bet[i]] += 1

for i in range(0,len(ver_arr_close)):
    counter_close[ver_arr_close[i]] += 1

#plot
lists = sorted(counter_bet.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.figure(figsize=(20,10))
plt.plot(x, y)
plt.savefig('dist_centrality_betweeness.png', bbox_inches='tight')
plt.show()

lists = sorted(counter_close.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.figure(figsize=(20,10))
plt.plot(x, y)
plt.savefig('dist_centrality_closeness.png', bbox_inches='tight')
plt.show()
