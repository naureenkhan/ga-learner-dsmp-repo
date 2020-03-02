# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# Load Offers
offers = pd.read_excel(path, sheet_name=0)
# Load Transactions
transactions = pd.read_excel(path, sheet_name=1)
# Merge dataframes
#transactions.columns = ["customer_name", "offer_id"]
transactions['n'] = 1
#transactions.insert("n",[1], True) 
df = offers.merge(transactions)
# Look at the first 5 rows
df.head()


# --------------
# Code starts here

# create pivot table
matrix = df.pivot_table(index='Customer Last Name', columns='Offer #', values='n')

# replace missing values with 0
matrix.fillna(0, inplace=True)
matrix.reset_index(inplace=True)
matrix.head()
# reindex pivot table


# display first 5 rows


# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans
import pandas as pd 
# Code starts here

# initialize KMeans object

cluster = KMeans(n_clusters=5,init='k-means++', max_iter=300, n_init=10, random_state=0)
center = cluster.fit_predict(matrix[matrix.columns[1:]]) 
print(center)
matrix['cluster'] = center
matrix.head()

# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca = PCA(n_components=2,random_state=0)
matrix['x']= pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y']= pca.fit_transform(matrix[matrix.columns[1:]])[:,1]

# dataframe to visualize clusters by customer names
clusters = matrix.iloc[:,[0,33,34,35]]
# visualize clusters
clusters.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')

# Code ends here


# --------------
# Code starts here
#
data = ['clusters','transactions']
# merge 'clusters' and 'transactions'
data = clusters.merge(transactions)

# merge `data` and `offers`
data = pd.merge(offers,data)
# initialzie empty dictionary
champagne= {}

# iterate over every cluster
for i in range(0,5):
    # observation falls in that cluster
    new_df = data[data['cluster']==i]
    # sort cluster according to type of 'Varietal'
    counts = new_df['Varietal'].value_counts(ascending=False)
    # check if 'Champagne' is ordered mostly
    if counts.index[0] == 'Champagne':
        champagne[i] =(counts[0])
# add it to 'champagne'
cluster_champagne=max(champagne , key = champagne.get)
print(cluster_champagne)

# get cluster with maximum orders of 'Champagne' 


# print out cluster number




# --------------
# Code starts here

# empty dictionary
discount = {}

# iterate over cluster numbers
for i in range(0,5):

    # dataframe for every cluster
    new_df=data[data.cluster == i]

    # average discount for cluster
    counts = new_df['Discount (%)'].values.sum()/len(new_df)
    # adding cluster number as key and average discount as value 
    discount[i] = counts

# cluster with maximum average discount
cluster_discount = max(discount,key=discount.get)
print(cluster_discount)
# Code ends here


