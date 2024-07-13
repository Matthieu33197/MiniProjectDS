import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Préparation de la donnée pour machine learning
# Séparer les colonnes numériques et non numériques
numeric_cols = train.select_dtypes(include=[np.number]).columns.drop('SalePrice')
non_numeric_cols = train.select_dtypes(exclude=[np.number]).columns

# Remplissage des données manquante par les médians et les modes
train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].median())
test[numeric_cols] = test[numeric_cols].fillna(test[numeric_cols].median())
train[non_numeric_cols] = train[non_numeric_cols].fillna(train[non_numeric_cols].mode().iloc[0])
test[non_numeric_cols] = test[non_numeric_cols].fillna(test[non_numeric_cols].mode().iloc[0])

# Encoder les colonnes non numériques en colonnes binaires
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Aligner les colonnes de train et test en supprimant les colonnes supplémentaire de test et en ajoutant à test les colonnes supplémentaires de train
train, test = train.align(test, join='left', axis=1)
test.fillna(0, inplace=True)

# Séparation des caractéristiques et la cible
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Division des données en deux parties -> Entrainement et validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation depuis l'écart type et la moyenne
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test = test.drop('SalePrice', axis=1, errors='ignore')
test = scaler.transform(test)