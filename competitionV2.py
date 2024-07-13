import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Préparation de la donnée pour machine learning
# Séparer les colonnes numériques et non numériques
numeric_cols = train.select_dtypes(include=[np.number]).columns.drop('SalePrice')
non_numeric_cols = train.select_dtypes(exclude=[np.number]).columns

# Remplissage des données manquantes par les médianes et les modes
train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].median())
test[numeric_cols] = test[numeric_cols].fillna(test[numeric_cols].median())
train[non_numeric_cols] = train[non_numeric_cols].fillna(train[non_numeric_cols].mode().iloc[0])
test[non_numeric_cols] = test[non_numeric_cols].fillna(test[non_numeric_cols].mode().iloc[0])

# Encoder les colonnes non numériques en colonnes binaires
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Aligner les colonnes de train et test en supprimant les colonnes supplémentaires de test et en ajoutant à test les colonnes supplémentaires de train
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
X_test = test.drop('SalePrice', axis=1, errors='ignore')
X_test = scaler.transform(X_test)

# Entraîner un modèle de régression random forest
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Calcule mse + r2 pour voir si notre modèle a bien appris
y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Faire des prédictions sur l'ensemble de test
y_test_pred = model.predict(X_test)

# Préparer le fichier de soumission
submission = pd.DataFrame({'Id': test.index, 'SalePrice': y_test_pred})
submission.to_csv('submission.csv', index=False)
