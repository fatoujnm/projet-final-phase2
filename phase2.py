# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
df = pd.read_csv('AmesHousing.csv')

# Nettoyage des données
df.columns = df.columns.str.strip()
df.fillna(df.median(numeric_only=True), inplace=True)
df['Year Built'] = pd.to_numeric(df['Year Built'], errors='coerce')

# Calcul de l'âge de la maison
df['Age'] = df['Yr Sold'] - df['Year Built']

# Encodage des variables catégorielles
df_encoded = pd.get_dummies(df, columns=['Neighborhood', 'House Style'])

# Visualisation de la distribution de SalePrice
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, bins=50)
plt.title('Distribution des Prix de Vente')
plt.xlabel('Prix de Vente')
plt.ylabel('Fréquence')
plt.show()

# Relation entre Gr Liv Area et SalePrice
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Gr Liv Area'], y=df['SalePrice'])
plt.title('Surface Habitable par Rapport au Prix de Vente')
plt.xlabel('Surface Habitable (pieds carrés)')
plt.ylabel('Prix de Vente')
plt.show()

# Relation entre Total Bsmt SF et SalePrice
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Total Bsmt SF'], y=df['SalePrice'])
plt.title('Surface Totale du Sous-Sol par Rapport au Prix de Vente')
plt.xlabel('Surface du Sous-Sol (pieds carrés)')
plt.ylabel('Prix de Vente')
plt.show()

# Relation entre 1st Flr SF et SalePrice
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['1st Flr SF'], y=df['SalePrice'])
plt.title('Surface du Premier Étage par Rapport au Prix de Vente')
plt.xlabel('Surface du Premier Étage (pieds carrés)')
plt.ylabel('Prix de Vente')
plt.show()

# Matrice de corrélation
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Matrice de Corrélation des Variables Numériques')
plt.show()

# Préparation des données pour le modèle de régression
# Sélection des variables
X = df[['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF']]
y = df['SalePrice']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Erreur Quadratique Moyenne (MSE) : {mse}')
print(f'Coefficient de Détermination (R²) : {r2}')

