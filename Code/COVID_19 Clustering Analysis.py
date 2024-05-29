import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from functools import reduce
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', 400)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')


# Set up the working directory
os.getcwd()

#===================================================================================
# Data Handling & Cleaning

data = pd.read_csv('Data/Aggregate_Data/dataset.csv')
data = data.drop([ 'std_fox_rtg', 'std_cnn_rtg', 'std_msn_rtg', 'std_local_rtg', 'twt501'], axis = 1)
data = data.dropna()

# Remove non-numeric and irrelevant columns
columns_to_remove = ['area_name', 'State', 'pop_cat', 'Area_Type']
data_cleaned = data.drop(columns=columns_to_remove)

# Remove rows with missing values
data_cleaned = data_cleaned.dropna()

column_name_mapping = {
    'FIPS': 'DE-FI',
    'MMR_VR': 'HE-MV',
    'pop2020': 'DE-PO',
    'VH_2021_12_15': 'HE-VH',
    'VH_2021_09_16': 'HE-VS',
    'case_rate_9_16_21': 'HE-CS',
    'case_rate_12_15_21': 'HE-CD',
    'black_pct': 'DE-BP',
    'hispanic_pct': 'DE-HP',
    'higher_edu': 'SO-HE',
    'med_hhincome': 'SO-MI',
    'median_age': 'DE-MA',
    'vehicle_perhh': 'SO-VH',
    'wo_insur_pct': 'HE-WI',
    'republican': 'PO-RE'
}

# Rename columns
data_cleaned.rename(columns=column_name_mapping, inplace=True)

data_cleaned_with_FIPS = data_cleaned.copy()
data_cleaned_without_FIPS = data_cleaned_with_FIPS.drop(columns = ['DE-FI'])

#===================================================================================
# Correlation Plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming data_cleaned_without_FIPS is already defined and cleaned
data_cor = data_cleaned_without_FIPS.copy()

# Compute the correlation matrix
corr_matrix = data_cor.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 10), dpi=300)

# Generate a custom diverging colormap
cmap = sns.color_palette("vlag_r", 7)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f", 
            annot_kws={"size": 10, "fontweight": "bold"},
            cbar_kws=dict(use_gridspec=True, location="right", shrink=0.5, ticks=[-1, -0.5, 0, 0.5, 1]))

# Customize plot appearance
# plt.title('Correlation Matrix of COVID-19 Data', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=18)
plt.yticks(rotation=0, fontsize=18)
plt.tight_layout()
plt.grid(False)
plt.legend(fontsize = 18)

plt.savefig('Figures/correlation_plot.pdf', format = 'pdf', dpi = 300)

# Show the plot
plt.show()


#===================================================================================
# Analysis 

## Data Scaling

data_scaled = data_cleaned_without_FIPS.copy()

## Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_scaled)

#===================================================================================
## Principal Component Analysis (PCA)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA()
pca.fit(data_scaled)

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance with annotations
plt.figure(figsize=(12, 8), dpi=300)
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o',
         linestyle='--', color='b', label='Cumulative Explained Variance', markersize=14)

# Annotate the plot
plt.xlabel('Number of Principal Components', fontsize=20)
plt.ylabel('Cumulative Explained Variance', fontsize=20)
#plt.title('Cumulative Explained Variance by Principal Components', fontsize=16, fontweight='bold')
plt.grid(True)

# Highlight the elbow point (assuming it's around the 5th component for demonstration)
elbow_point = 8
plt.axvline(x=elbow_point, color='r', linestyle='--')
plt.axhline(y=cumulative_explained_variance[elbow_point-1], color='r', linestyle='--')
plt.text(elbow_point-0.5, cumulative_explained_variance[elbow_point+2]-0.05, f'{cumulative_explained_variance[elbow_point-1]:.2f}', color='red', fontsize=12)

# Add a threshold line for 90% explained variance
threshold = 0.90
plt.axhline(y=threshold, color='g', linestyle='--', label='90% Threshold')
num_components_threshold = np.argmax(cumulative_explained_variance >= threshold) + 1
plt.axvline(x=num_components_threshold, color='r', linestyle='--')
plt.text(num_components_threshold+0.5, threshold-0.03, f'{threshold*100:.0f}% Variance', color='green', fontsize=12)

# Customize ticks and labels
plt.xticks( fontsize=18)
plt.yticks( fontsize=18)
plt.legend(loc='lower right', fontsize=18)
plt.tight_layout()

plt.savefig('Figures/cumulative_variance.pdf', format = 'pdf', dpi = 300)

# Show the plot
plt.show()

#===================================================================================
# Pick the top 8 variables

n_compontents = 8

pca = PCA(n_components=n_compontents)
pca.fit(data_scaled)

# Get the loadings (eigenvectors)
loadings = pca.components_.T

# Create a DataFrame for the loadings
loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(n_compontents)], index=data_cleaned_without_FIPS.columns)

# Sum the absolute loadings for each variable across the top 10 principal components
loading_sums = loadings_df.abs().sum(axis=1)

# Sort the variables by the sum of their absolute loadings
top_8_variables = loading_sums.sort_values(ascending=False).head(n_compontents)
top_8_variables


top_variables = dict(top_8_variables)
top_variables = list(top_variables.keys())


# Filter the dataset to include only the top variables

data_top_variables = data_cleaned_without_FIPS[top_variables]

# Standardize the data
data_scaled = scaler.fit_transform(data_top_variables)

#===================================================================================

## KMeans Clustering


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Calculate silhouette scores for different number of clusters
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Find the number of clusters with the highest silhouette score
best_n_clusters = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]
best_score = max(silhouette_scores)


# Plot the Silhouette Method
plt.figure(figsize=(12, 8), dpi=300)
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--', color='b', markersize=14)

# Annotate the best score
plt.annotate(f'Best: {best_n_clusters} clusters\nScore: {best_score:.2f}', 
             xy=(best_n_clusters, best_score), xycoords='data',
             xytext=(best_n_clusters, best_score-0.05), textcoords='data',
             arrowprops=dict(facecolor='black', shrink=0.02),
             horizontalalignment='right', verticalalignment='top', fontsize=18)

# Customize plot appearance
# plt.title('Silhouette Method Evaluation for KMeans Clustering', fontsize=16, fontweight='bold')
plt.xlabel('Number of Clusters', fontsize=20)
plt.ylabel('Silhouette Score', fontsize=20)
plt.xticks(range(2, 11), fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle='--', linewidth=0.5)

# Tight layout for better spacing
plt.tight_layout()

plt.savefig('Figures/silhouette_evaluation.pdf', format = 'pdf', dpi = 300)

# Show the plot
plt.show()


# Apply K-Means clustering with 4 clusters
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# Adjust cluster labels to start from 1 instead of 0
kmeans_labels = kmeans_labels + 1

# Add the cluster labels and FIPS column back to the original data
data_cleaned_with_FIPS['KMeans_Labels'] = kmeans_labels


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Evaluate K-Means clustering with 4 clusters
silhouette = silhouette_score(data_scaled, kmeans_labels)
davies_bouldin = davies_bouldin_score(data_scaled, kmeans_labels)
calinski_harabasz = calinski_harabasz_score(data_scaled, kmeans_labels)

print(f'Silhouette Score (4 Clusters): {silhouette}')
print(f'Davies-Bouldin Index (4 Clusters): {davies_bouldin}')
print(f'Calinski-Harabasz Index (4 Clusters): {calinski_harabasz}')


import itertools

df_pairplot = data_cleaned_with_FIPS.copy()
df_pairplot['Cluster Number'] = df_pairplot['KMeans_Labels']
# Generate pairplot
pairplot = sns.pairplot(df_pairplot, vars= top_variables, hue='Cluster Number', palette='coolwarm', diag_kind='kde', plot_kws={'alpha':1, 'edgecolor': 'none'})

for ax in pairplot.axes.flatten():
    ax.tick_params(labelsize=18)  # Update the tick font size
    ax.xaxis.label.set_size(20)   # Update the x-axis label font size
    ax.yaxis.label.set_size(20)   # Update the y-axis label font size

# Updating the legend font size
pairplot._legend.set_title('Cluster\nNumber')  # Set a title for the legend
pairplot._legend.get_title().set_fontsize(18)  # Update the legend title font size
for text in pairplot._legend.get_texts():
    text.set_fontsize(16)  # Update the legend labels font size

# Adjusting the marker sizes in the legend
for handle in pairplot._legend.legendHandles:
    handle._sizes = [150]  # Adjust the size of the markers
    
plt.savefig('Figures/kmeans_5_clusters_pairplot.pdf', format='pdf', dpi=300)
plt.show()


#===================================================================================

## Hierarchical Clustering

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Hierarchical Clustering
linkage_matrix = linkage(data_scaled, method='ward')
hierarchical_labels = fcluster(linkage_matrix, t=5, criterion='maxclust')

data_cleaned_with_FIPS['Hierarchical_Labels'] = hierarchical_labels


fig, ax = plt.subplots(figsize=(14, 8))
dendro = dendrogram(
    linkage_matrix, 
    truncate_mode='level', 
    p=5, 
    show_leaf_counts=True, 
    leaf_rotation=90., 
    leaf_font_size=10.,
    ax=ax
)

# Customize the colors and alpha
for i, d in zip(dendro['icoord'], dendro['dcoord']):
    x = 0.5 * sum(i[1:3])
    y = d[1]
    plt.plot(i, d, 'k-', color='blue', alpha=0.3)  # Change color and alpha
# plt.title('Dendrogram for Hierarchical Clustering', fontsize=16)
plt.xlabel('Cluster Size', fontsize=20)
plt.ylabel('Distance', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=18)
plt.grid(False)
plt.savefig('Figures/hierarchical_dendogram.pdf', format = 'pdf', dpi = 300)
plt.show()


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Evaluate K-Means clustering with 4 clusters
silhouette = silhouette_score(data_scaled, hierarchical_labels)
davies_bouldin = davies_bouldin_score(data_scaled, hierarchical_labels)
calinski_harabasz = calinski_harabasz_score(data_scaled, hierarchical_labels)

print(f'Silhouette Score (4 Clusters): {silhouette}')
print(f'Davies-Bouldin Index (4 Clusters): {davies_bouldin}')
print(f'Calinski-Harabasz Index (4 Clusters): {calinski_harabasz}')

pairplot = sns.pairplot(data_cleaned_with_FIPS, vars=top_variables, hue='Hierarchical_Labels', palette='coolwarm', diag_kind='kde', plot_kws={'alpha': 1, 'edgecolor': 'none'})

for ax in pairplot.axes.flatten():
    ax.tick_params(labelsize=18)  # Update the tick font size
    ax.xaxis.label.set_size(20)   # Update the x-axis label font size
    ax.yaxis.label.set_size(20)   # Update the y-axis label font size

# Updating the legend font size
pairplot._legend.set_title('Cluster\nNumber')  # Set a title for the legend
pairplot._legend.get_title().set_fontsize(18)  # Update the legend title font size
for text in pairplot._legend.get_texts():
    text.set_fontsize(16)  # Update the legend labels font size

# Adjusting the marker sizes in the legend
for handle in pairplot._legend.legendHandles:
    handle._sizes = [150]  # Adjust the size of the markers

plt.savefig('Figures/hierarchical_5_clusters.pdf', format = 'pdf', dpi = 300)

plt.show()


#===================================================================================

## Data Profiling

kmeans_profiles = data_cleaned_with_FIPS.groupby('KMeans_Labels').mean().transpose()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
df = data_cleaned_with_FIPS.copy()
df = df.drop(columns=['DE-FI', 'Hierarchical_Labels'])
subset_df = df.loc[:, ['HE-MV', 'DE-PO', 'HE-VS', 'HE-VH', 'HE-CS', 'HE-CD', 'DE-BP', 'KMeans_Labels']]

# List of variables to plot
variables = subset_df.columns[:-1]
clusters = sorted(subset_df['KMeans_Labels'].unique())

# Create a wide plot with subplots for each variable and each cluster
fig, axes = plt.subplots(nrows=len(clusters), ncols=len(variables), figsize=(20, 15), sharex=False, sharey=False)

for i, cluster in enumerate(clusters):
    cluster_data = subset_df[subset_df['KMeans_Labels'] == cluster]
    for j, var in enumerate(variables):
        if not cluster_data.empty:
            sns.kdeplot(data=cluster_data, x=var, ax=axes[i, j], fill=True, color='lightblue', alpha=0.75)
            
            # Calculate min, median, and max
            min_val = cluster_data[var].min()
            median_val = cluster_data[var].median()
            max_val = cluster_data[var].max()
            
            # Add vertical lines
            axes[i, j].axvline(min_val, color='blue', linestyle='--', linewidth=1)
            axes[i, j].axvline(median_val, color='green', linestyle='-', linewidth=2)
            axes[i, j].axvline(max_val, color='red', linestyle='--', linewidth=1)

        if i == 0:
            axes[i, j].set_title(var, fontsize=18)
        if j == 0:
            axes[i, j].set_ylabel(f'Cluster {cluster}', fontsize=16)
        else:
            axes[i, j].set_ylabel('')
        axes[i, j].set_xlabel('')
        axes[i, j].set_yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('Figures/density_plot_first_half.pdf', format='pdf', dpi=300)

plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
df = data_cleaned_with_FIPS.copy()
df = df.drop(columns = ['DE-FI', 'Hierarchical_Labels'])
subset_df = df.loc[:, [ 'DE-HP', 'SO-HE', 'SO-MI', 'DE-MA', 'SO-VH', 'HE-WI', 'PO-RE', 'KMeans_Labels']]

# List of variables to plot
variables = subset_df.columns[:-1]
clusters = sorted(subset_df['KMeans_Labels'].unique())

# Create a wide plot with subplots for each variable and each cluster
fig, axes = plt.subplots(nrows=len(clusters), ncols=len(variables), figsize=(20, 15), sharex=False, sharey=False)

for i, cluster in enumerate(clusters):
    cluster_data = subset_df[subset_df['KMeans_Labels'] == cluster]
    for j, var in enumerate(variables):
        if not cluster_data.empty:
            sns.kdeplot(data=cluster_data, x=var, ax=axes[i, j], fill=True, color='lightblue', alpha=0.75)
            
            # Calculate min, median, and max
            min_val = cluster_data[var].min()
            median_val = cluster_data[var].median()
            max_val = cluster_data[var].max()
            
            # Add vertical lines
            axes[i, j].axvline(min_val, color='blue', linestyle='--', linewidth=1)
            axes[i, j].axvline(median_val, color='green', linestyle='-', linewidth=2)
            axes[i, j].axvline(max_val, color='red', linestyle='--', linewidth=1)

        if i == 0:
            axes[i, j].set_title(var, fontsize=18)
        if j == 0:
            axes[i, j].set_ylabel(f'Cluster {cluster}', fontsize=16)
        else:
            axes[i, j].set_ylabel('')
        axes[i, j].set_xlabel('')
        axes[i, j].set_yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('Figures/density_plot_second_half.pdf', format='pdf', dpi=300)

plt.show()




# Define the categories based on quantiles
def categorize_variable(column):
    quantiles = column.quantile([0.33, 0.67])
    categories = pd.cut(column, bins=[-np.inf, quantiles[0.33], quantiles[0.67], np.inf], labels=['Low', 'Medium', 'High'])
    return categories

# Apply the categorization to each variable (row)
categorized_profiles = kmeans_profiles.apply(categorize_variable, axis=1)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

## Heatmap
# Scale the DataFrame using MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

df_scaled = df_scaled.drop(columns = ['DE-FI', 'Hierarchical_Labels'])

# Plot the heatmap
plt.figure(figsize=(20, 12))
#ax = sns.heatmap(df_scaled, cmap='RdBu', square=True, linewidths=.5)

#mask = np.triu(np.ones_like(df_scaled, dtype=bool))
#cmap = sns.color_palette("vlag_r", 50000) 
cmap = sns.color_palette("RdBu", 13) 

# Draw the heatmap with the mask and correct aspect ratio
ax = sns.heatmap(df_scaled, cmap=cmap, vmax=1 , vmin = .8,  center=0.80,
            square=True, linewidths=.5, 
            cbar_kws = dict(use_gridspec=True,location="right",
                                                       shrink = 0.3))

plt.legend(loc='upper center')


# Customize the plot
# plt.xlabel('Variable', fontsize=20)
plt.ylabel('Cluster Number', fontsize=20)
plt.xticks(rotation=45, ha = 'right', fontsize=18)
plt.yticks(rotation=0, ha='right', fontsize=18)
#plt.tight_layout()

plt.savefig('Figures/heatmap_scaled_Kmeans.pdf', format = 'pdf', dpi = 300)

plt.show()


#===================================================================================

## Multiple Linear Regression

data_regression = data_cleaned_with_FIPS.copy()

KMeans_Labels = data_regression[['KMeans_Labels']]

scaler = MinMaxScaler()

data_regression = data_regression.drop(columns = ['DE-FI', 'Hierarchical_Labels', 'KMeans_Labels'])
data_regression = pd.DataFrame(scaler.fit_transform(data_regression), columns=data_regression.columns, index=data_regression.index)

data_regression['KMeans_Labels'] = KMeans_Labels


data_regression.to_csv('COVID_Clustering_data_regression.csv')































