# Question Link: https://www.hackerearth.com/problem/approximate/davies-bouldin-index/

import sys
sys.stdin=open('input.txt','r')


import pandas as pd
import numpy as np
import math
from sklearn.cluster import k_means
from scipy.spatial import distance

m = int(input())
n = int(input())
k = int(input())
data = []
for i in range(m):
    data += [[float(x) for x in raw_input().split() ]]
clf = k_means(data,n_clusters=k)

data = np.array(data)
centroids = clf[0]
labels = clf[1]

s_i = np.array([0.0]*k)
count_i = np.array([0]*k)
for i in range(m):
    s_i[ labels[i] ] += np.linalg.norm(data[i]-centroids[ labels[i] ])
    count_i[ labels[i] ] += 1
for i in range(k):
    s_i[i] /= count_i[i]

r_i = 0.0
for i in range(k):
    r_ij = []
    for j in range(k):
        if i!=j:
            r_ij += [ (s_i[i]+s_i[j])/np.linalg.norm(centroids[i]-centroids[j]) ]
    r_i += max(r_ij)
print("%.20f"%(r_i/k))
