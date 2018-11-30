import matplotlib.pyplot as plt
import pickle
import numpy as np


with open('count_clust_1mil.pkl','rb') as f:
	counter_clustering = pickle.load(f)

lists = sorted(counter_clustering.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.figure(figsize=(20,10))
plt.xlim([0,0.00005])
plt.plot(x, np.log10(y))
plt.savefig('dist_centrality_clustering.png', bbox_inches='tight')
#plt.show()

