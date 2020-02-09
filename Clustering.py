#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""group a set of data points (e.g. representing customers of a cafe) into coherent groups (clusters or segments)
using clustering methods. use hard clustering method k-means """


# In[1]:


##########Loading the Data.
# import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np

#read in data from the csv file and store it in the numpy array data.
df = pd.read_csv("data.csv")
data = np.array(df)

#display first 5 rows
display(df.head(5))  


# In[ ]:


def plotting(data, centroids=None, clusters=None):
    #this function will later on be used for plotting the clusters and centroids. But now we use it to just make a scatter plot of the data
    #Input: the data as an array, cluster means (centroids), cluster assignemnts in {0,1,...,k-1}   
    #Output: a scatter plot of the data in the clusters with cluster means
    plt.figure(figsize=(5.75,5.25))
    data_colors = ['orangered','dodgerblue','springgreen']
    plt.style.use('ggplot')
    plt.title("Data")
    plt.xlabel("feature $x_1$: customers' age")
    plt.ylabel("feature $x_2$: money spent during visit")

    alp = 0.5             # data points alpha
    dt_sz = 20            # marker size for data points 
    cent_sz = 130         # centroid sz 
    
    if centroids is None and clusters is None:
        plt.scatter(data[:,0], data[:,1],s=dt_sz,alpha=alp ,c=data_colors[0])
    if centroids is not None and clusters is None:
        plt.scatter(data[:,0], data[:,1],s=dt_sz,alpha=alp, c=data_colors[0])
        plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=cent_sz, c=centroid_colors[:len(centroids)])
    if centroids is not None and clusters is not None:
        plt.scatter(data[:,0], data[:,1], c=[data_colors[i-1] for i in clusters], s=dt_sz, alpha=alp)
        plt.scatter(centroids[:,0], centroids[:,1], marker="x", c=centroid_colors[:len(centroids)], s=cent_sz)
    if centroids is None and clusters is not None:
        plt.scatter(data[:,0], data[:,1], c=[data_colors[i-1] for i in clusters], s=dt_sz, alpha=alp)
    
    plt.show()

#plot the data
plotting(data)   


# In[ ]:


#####################  k-means   #####################
"The code snippet below uses k-means to cluster the customers into  𝑘=3  clusters "


from sklearn.cluster import KMeans

X = np.zeros([400,2])               # read the customer data into X
cluster_means = np.zeros([2,2])     # store the resulting clustering means in the rows of this np array
cluster_indices = np.zeros([400,1]) # store here the resulting cluster indices (one for each data point)
data_colors = ['orangered','dodgerblue','springgreen'] # colors for data points
centroid_colors = ['red','darkblue','limegreen'] # colors for the centroids

X = data      # read in  customer data point into numpy array X of shape (400,2)
k_means = KMeans(n_clusters = 3, max_iter = 100).fit(X) # apply k-means with k=3 cluster and using 100 iterations
cluster_means = k_means.cluster_centers_         # read out cluster means (centers)
cluster_indices = k_means.labels_                # read out cluster indices for each data point
cluster_indices = cluster_indices.reshape(-1,1)  # enforce numpy array cluster_indices having shape (400,1)

# code below creates a colored scatterplot 

plt.figure(figsize=(5.75,5.25))
plt.style.use('ggplot')
plt.title("Data")
plt.xlabel("feature $x_1$: customers' age")
plt.ylabel("feature $x_2$: money spent during visit")

alp = 0.5             # data points alpha
dt_sz = 40            # marker size for data points 
cent_sz = 130         # centroid sz 
       
# iterate over all cluster indices (minus 1 since Python starts indexing with 0)

for cluster_index in range(3):
    # find indices of data points which are assigned to cluster with index (cluster_index+1)
    indx_1 = np.where(cluster_indices == cluster_index)[0] 

    # scatter plot of all data points in cluster with index (cluster_index+1)
    plt.scatter(X[indx_1,0], X[indx_1,1], c=data_colors[cluster_index], s=dt_sz, alpha=alp) 
    

# plot crosses at the locations of cluster means 

plt.scatter(cluster_means[:,0], cluster_means[:,1], marker="x", c='black', s=cent_sz)
    
plt.show()


# In[ ]:


#####################  k-means   #####################
" Try out different number  𝑘  of clusters to select the best parameter"

data_num = data.shape[0]
err_clustering = np.zeros([8,1])

for k in range(8):
    k_means=KMeans(n_clusters=k+1,max_iter=100).fit(X)
    err_clustering[k]=k_means.inertia_/data_num  

fig=plt.figure(figsize=(8,6))
plt.plot(range(1,9),err_clustering)
plt.xlabel('Number of clusters')
plt.ylabel('Clustering error')
plt.title("The number of clusters vs clustering error")
plt.show()    


# In[2]:


#####################  k-means   #####################
"Handling Local Minima,repeat  𝑘 -means To Escape Local Minima. "

min_ind= 0  # store here the index of the repetition yielding smallest clustering error 
max_ind= 0  # .... largest clustering error

# initializing the array where we collect all cluster assignments  
cluster_assignment = np.zeros((50, data.shape[0]),dtype=np.int32)

clustering_err = np.zeros([50,1]) # init numpy array for storing the clustering errors in each repetition

np.random.seed(42) 

init_means_cluster1 = np.random.randn(50,2)  # use the rows of this numpy array to init k-means 
init_means_cluster2 = np.random.randn(50,2)  # use the rows of this numpy array to init k-means 
init_means_cluster3 = np.random.randn(50,2)  # use the rows of this numpy array to init k-means 

best_assignment = np.zeros((400,1))     # store here the cluster assignment achieving smallest clustering error
worst_assignment = np.zeros((400,1))    # store here the cluster assignment achieving largest clustering error

init_means_cluster_ar=np.zeros((50,3,2))

for i in range(50):
    init_means_cluster_ar[i]=[init_means_cluster1[i,:],init_means_cluster2[i,:],init_means_cluster3[i,:]]

data_num = data.shape[0]

for i in range(50):
    k_means=KMeans(n_clusters = 3,init=init_means_cluster_ar[i],max_iter = 100).fit(X)
    err_clustering=k_means.inertia_/data_num
    clustering_err[i]=err_clustering
    cluster_assignment[i]=k_means.labels_

min_ind=np.argmin(clustering_err)
best_assignment=cluster_assignment[min_ind]

max_ind=np.argmax(clustering_err)
worst_assignment=cluster_assignment[max_ind]

print("Cluster assignment with smallest clustering error")
plotting(data, clusters = cluster_assignment[min_ind, :])
print("Cluster assignment with largest clustering error")
plotting(data, clusters = cluster_assignment[max_ind,:])


# In[3]:


###################### Density Based Clustering

"""Density-based spatial clustering of applications with noise (DBSCAN) is a hard clustering method that uses a connectivity
based similarity measure. In contrast to k-means and GMM, DBSCAN does not require the number of clusters as an argument; the 
number of clusters used depends its parameters. Moreover, DBSCAN allows to detect outliers which can be interpreted as degenerated 
clusters consisting of exactly one data point. For a detailed discussion of how DBSCAN works, 
we refer to https://en.wikipedia.org/wiki/DBSCAN
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph

np.random.seed(844)
clust1 = np.random.normal(5, 2, (1000,2))
clust2 = np.random.normal(15, 3, (1000,2))
clust3 = np.random.multivariate_normal([17,3], [[1,0],[0,1]], 1000)
clust4 = np.random.multivariate_normal([2,16], [[1,0],[0,1]], 1000)

dataset1 = np.concatenate((clust1, clust2, clust3, clust4))

# we take the first array as the second array has the cluster labels
dataset2 = datasets.make_circles(n_samples=1000, factor=.5, noise=.05)[0]

# plot clustering output on the two datasets
def cluster_plots(set1, set2, colours1, colours2, 
                  title1 = 'Dataset 1',  title2 = 'Dataset 2'):
    colours1 = colours1.reshape(-1,)
    colours2 = colours2.reshape(-1,)
    fig,(ax1,ax2) = plt.subplots(1, 2)
    fig.set_size_inches(6, 3)
    ax1.set_title(title1,fontsize=14)
    ax1.set_xlim(min(set1[:,0]), max(set1[:,0]))
    ax1.set_ylim(min(set1[:,1]), max(set1[:,1]))
    ax1.scatter(set1[:, 0], set1[:, 1],s=8,lw=0,c=colours1)
    ax2.set_title(title2,fontsize=14)
    ax2.set_xlim(min(set2[:,0]), max(set2[:,0]))
    ax2.set_ylim(min(set2[:,1]), max(set2[:,1]))
    ax2.scatter(set2[:, 0], set2[:, 1],s=8,lw=0,c=colours2)
    fig.tight_layout()
    plt.show()


# implementing DBSCAN
from sklearn.cluster import DBSCAN

dbscan_dataset1=DBSCAN(eps=1,min_samples=5,metric='euclidean').fit_predict(dataset1).reshape(-1,1)

dbscan_dataset1_sel=np.select([dbscan_dataset1==-1],[1],default=0)
dataset1_noise_points=int(sum(dbscan_dataset1_sel))

dbscan_dataset2=DBSCAN(eps=0.1,min_samples=5,metric='euclidean').fit_predict(dataset2).reshape(-1,1)
dbscan_dataset2_sel=np.select([dbscan_dataset2==-1],[1],default=0)
dataset2_noise_points=int(sum(dbscan_dataset2_sel))

print(dbscan_dataset1)

print(dbscan_dataset1.shape)
print('Dataset1:')
print("Number of Noise Points: ",dataset1_noise_points," (",len(dbscan_dataset1),")",sep='')
print('Dataset2:')
print("Number of Noise Points: ",dataset2_noise_points," (",len(dbscan_dataset2),")",sep='')

cluster_plots(dataset1, dataset2, dbscan_dataset1, dbscan_dataset2)


# In[ ]:




