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