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
plt.figure(figsize=(20, 16)) 
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, annot_kws={"size": 10})
plt.title('Matrice de Corrélation')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.show()
