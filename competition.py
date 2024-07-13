import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Préparation de la donnée
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Affichage histogramme
plt.figure(figsize=(10, 6))
sns.histplot(train['SalePrice'], kde=True)
plt.title('Distribution des Prix de Vente')
plt.xlabel('Prix de Vente')
plt.ylabel('Fréquence')
plt.show()

# Heatmap des corrélations
numeric_cols = train.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()
corr_matrix = corr_matrix[(corr_matrix != 1.000) & ((corr_matrix > 0.5) | (corr_matrix < -0.2))]
plt.figure(figsize=(20, 16)) 
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, annot_kws={"size": 10})
plt.title('Matrice de Corrélation')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# Scatter plot de quelques caractéristiques importantes
important_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
axes = axes.flatten()
for i, feature in enumerate(important_features):
    sns.scatterplot(data=train, x=feature, y='SalePrice', ax=axes[i])
    axes[i].set_title(f'{feature} vs Prix de Vente')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Prix de Vente')
plt.tight_layout()
plt.show()