import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv('Cleaned-Data.csv')
print(df.head())
del df['Country']
print(df.describe())
df.drop_duplicates()

# Exploratory data analysis

figure = plt.figure(figsize=(30, 30))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Preparing data for analysis
print(df.columns)

train_df = df.copy()
# scaling data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_df_scaled = scaler.fit_transform(train_df)

inertias = []

for i in range(1, 15):
    km = KMeans(n_clusters=i, random_state=2)
    km.fit(train_df_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 15), inertias, color='blue', marker='o', markerfacecolor='red', markersize=10)
plt.title('Inertias vs. number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

from sklearn.metrics import davies_bouldin_score
bouldin_score=[]

for i in range(4,15):
    km=KMeans(n_clusters=i, random_state=2)
    labels=km.fit_predict(train_df_scaled)
    bouldin_score.append(davies_bouldin_score(train_df_scaled, labels))
plt.figure(figsize=(10,6))
plt.plot(range(4,15), bouldin_score, color='blue', marker='o', markerfacecolor='red', markersize=10)
plt.title('Davies Bouldin Score vs. number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Davies Bouldin Score')
plt.show()

km_model=KMeans(n_clusters=7, random_state=2)
km_model.fit(train_df_scaled)

labels=km_model.labels_

corona_df=pd.DataFrame(km_model.cluster_centers_, columns=train_df.columns)
print(corona_df)

covid_pca=PCA(n_components=3)
principal_comp=covid_pca.fit_transform(train_df_scaled)
principal_comp=pd.DataFrame(principal_comp,columns=['pca1','pca2','pca3'])
print(principal_comp)

principal_comp1=pd.concat([principal_comp,pd.DataFrame({"Cluster":labels})],axis=1)
print(principal_comp1.sample(10))

plt.figure(figsize=(10,10))
ax=sns.scatterplot(x='pca1',y='pca2',hue="Cluster",data=principal_comp1, palette=['red', 'blue', 'yellow', 'black', 'orange', 'green', 'violet'])
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
sc=ax.scatter(xs=principal_comp1['pca1'],ys=principal_comp1['pca3'],zs=principal_comp1['pca2'],c=principal_comp1['Cluster'],marker='o',cmap="gist_rainbow")
plt.colorbar(sc)
plt.show()