import pandas as pd
import numpy as np
import matplotlib 
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score , f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings




def pipeline_nettoyage_modele(df_training):

    # On identifie les colonnes automatiquement
    num_cols = df_training.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df_training.select_dtypes(include=['object']).columns

    #on gere le nettoyage des chiffres
    num_transformer =Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]) 

    #On gere les textes ou les variables categorielles
    cat_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])

    #on combine le tout dans un prepocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
    ])


    #Pipeline nettoyage + modele pour le j'ai choisi Gradient() je le change en fonction du projet
    modele_pipeline = Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('regressor',KNeighborsClassifier(n_neighbors=5, weights='distance'))

    ])

    return modele_pipeline




#Gestion des outliers par la methode IQR

def cleaner_outlier(df, colum):


         # On vérifie donc ici s'il a bien l'attribut 'columns'
    if not hasattr(df, 'columns'):
        return df # Si c'est une Series, on ne fait rien pour éviter le crash
       
    #verifie si la colonne existe dans le dataframe
    if colum not in df.columns:
         print(f"Attention : la colonne {colum} est absente du dataframe")

    Q1 = df[colum].quantile(0.25)
    Q3 = df[colum].quantile(0.75)
    IQR = Q3 - Q1

    #limiter les extremites
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    # On crée un masque : vrai si la valeur est dans les bornes
    masque = (df[colum] >= lower_bound) & (df[colum] <= upper_bound)
    #df_out = df[masque]

    return df[masque].copy()


def compare_modele(X_train,X_test,y_train,y_test, pipeline_base):

    modeles = {
    'logistic regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}
    #on recupere seulement la partie nettoyage
    preprocessing = pipeline_base.steps[0][1]

    scores = []

    for nom, model_brut in modeles.items():

        #j'utilise le pretraitement que le modele_nettoyage en haut
       
        mon_pipe = make_pipeline(preprocessing, model_brut)

        mon_pipe.fit(X_train, y_train)
        score = mon_pipe.score(X_test, y_test)
        scores.append({'modele':nom, 'Score R2':score})

        
    return pd.DataFrame(scores).sort_values(by="Score R2", ascending=False)



#la fonction qui verifie l'asymetrie ou la symetrie des donnees numeriques

def asymetrique_symetrique(df, colonne):

    plt.figure(figsize=(8 , 5))

   #calcule la symetrie(skewness)
    val_sym = df[colonne].skew()

    #histogramme + courbe de densite
    sns.histplot(df[colonne], kde=True, color='teal', alpha=0.5)

    #placement des moyenne et mediane dans le repere pour comparer
    #si les deux courbes sont colees(moyenne) sinon(mediane)
    moyenne = df[colonne].mean()
    mediane = df[colonne].median()

    #on trace une ligne verticale pour voir les deux lignes
    plt.axvline(moyenne, color='red', linestyle='--', label = f"Moyenne: {moyenne:.2f}.")
    plt.axvline(mediane, color='green', linestyle='-', label = f"Mediane : {mediane:.2f}")

    plt.title(f"Distribution de {colonne} (Skewness: {val_sym:.2f})")
    plt.legend()
    plt.show() 

    #return df




##fonction pour le pretraitement des donnees (valeurs manquantes, stats descrp etc...)

def pretraitement_data(df):

    #gere des colonnes trop vides 
    miss = df.isnull().sum()
    miss_percent = (miss / len(df)) * 100
    cols_supp = miss_percent[miss_percent > 80].index
    df = df.drop(columns=cols_supp)
    print(f"colonnes supprimer (>80%) de vides : {list(cols_supp)}")

    #separation automatiques des types
    col_num = df.select_dtypes(include=[np.number]).columns
    col_cat = df.select_dtypes(include=['object']).columns

    #on va fiare un boucle pour les colonnes numeriques
    for col in col_num:
        if df[col].isnull().sum() > 0:   #les valeurs manquantes
             symetrie = df[col].skew()

             if abs(symetrie) < 0.5:
                 #symetrie : moyenne
                 df[col] = df[col].fillna(df[col].mean())
                 print(f" -{col} : Moyenne utilise (skew : {symetrie:.2f})")
             else:
                df[col] = df[col].fillna(df[col].median())
                print(f" -{col} : Mediane utilise (skew : {symetrie:.2f})")

    #remplissage des valeurs categorielles par leur mode
    for cat in col_cat:
        if df[cat].isnull().sum() > 0:
            #on calcule le mode
            modes = df[cat].mode()
            #on verifie si la colonne n'est pas vide
            if not modes.empty:
                valeur_rempli = modes.iloc[0]
                if isinstance(valeur_rempli, (list, dict)):
                    valeur_rempli = str(valeur_rempli)
                df[cat] = df[cat].fillna(valeur_rempli)
           # df[cat] = df[cat].fillna(df[cat].mode()[0])
                print(f"  -{cat} : mode utilise")
            else:
                 print(f"  -{cat} : impossible colonne vide")

   # print("pretraitement terminer avec succes!!!")

    return df


#Faire le graphique boxplot pour verifier les outliers
def verifier_outlier(df, colonne):

    plt.figure(figsize=(8, 2))
    sns.boxplot(x = df[colonne], color='salmon')
    plt.title(f"verification des outliers : {colonne}")
    plt.show()



#Fonction généralisée pour évaluer une classification binaire ou multiclasse.
#Prend en compte les prédictions de régression en les arrondissant.

def evaluer_classification(y_reel, y_pred, nom_model='modele'):

    #gerer les predictions continues pour au lieu de renvoyer une probabilite avce np.round ça renvoi(0 ou 1) pour calculer les metriques
    y_pred_class = np.round(y_pred)  

    #calcule des metriques
    acc = accuracy_score(y_reel, y_pred_class)
    f1 = f1_score(y_reel, y_pred_class)
    prec = precision_score(y_reel, y_pred_class)
    rec = recall_score(y_reel, y_pred_class)

    #Affichage des metriques

    print(f"----Evaluation du: {nom_model}----")
    print(f" Precision (Accuracy) : {acc * 100:.2f}%")
    print(f"F1 score :{f1 * 100:.2f}%")
    print(f"Precision :{prec * 100:.2f}%")
    print(f"recall : {rec * 100:.2f}%")
    print("-" * 30)

    #Matrice de confusion
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_reel, y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',cbar=False)
    plt.title("Matrice de confusion")
    plt.xlabel("Predictions")
    plt.ylabel("Realite")

    plt.show()


    #Fonction cross-validation ou validation croisee

def valider_stabilite(modele, X, y, cv=5, metrique='accuracy'):

        #calcule de score sur 5 decoupages
    scores = cross_val_score(modele, X, y, cv=cv, scoring=metrique)

    moyenne= scores.mean()
    ecart_type = scores.std()

        # 3. Affichage propre
    print(f"--- VALIDATION CROISÉE ({cv} Folds) ---")
    print(f"Moyenne des scores ({metrique}) : {moyenne:.2f}%")
    print(f"Écart-type (instabilité)     : +/- {ecart_type:.2f}%")

    if ecart_type > 5:
        print("le modele est instable")
    else:
        print("le modele est stable et fiable")


    return scores  

#J'ai choisit le seuil de 5% car au delà, la variabilite des performence suggere que le modele est trop sensible
# au donnees d'entrainement(overfiting)    

#------------------------------------------------
#Une fonction pour les hyperparametres

#Cherche la meilleure combinaison d'hyperparamètres pour un algorithme donné.
#Retourne le meilleur modèle déjà entraîné.
    
def optimiser_modele(algo, parametre, X, y, cv=5):

    print("Optimisation :")
    grid= GridSearchCV(estimator=algo, param_grid=parametre, cv=cv, scoring='accuracy', n_jobs=1, verbose=1)

    #Execute les test
    grid.fit(X, y)

    #Affichage des resultats
    print(f"meilleur parametre : {grid.best_params_}")
    print(f"meilleur score :{grid.best_score_ * 100 :.2f}")


    return grid.best_estimator_


##Super optimisateur
def super_optimisateur(pipeline, grille_params, X_train, y_train):
    
    #on ignore les avertissements techniques
    warnings.filterwarnings('ignore')

    #Configuration par GridSearch
    grid =GridSearchCV(
        estimator=pipeline,
        param_grid=grille_params,
        cv=5,
        scoring='accuracy',
        verbose=1 #Affiche la progression
    )

    #Entrainement
    grid.fit(X_train, y_train)

    # 4. Affichage des résultats
    print("-" * 30)
    print(f"MEILLEUR SCORE : {grid.best_score_:.4f}")
    print(f"MEILLEURS PARAMÈTRES : {grid.best_params_}")
    print("-" * 30)
    
    # Renvoie le meilleur modèle déjà entraîné
    return grid.best_estimator_


#Des fonctions pour extraire les donnees detaillees via Ids dans les APIs
#pour les rockets/fusee
import requests

def getBoosterVersion(data, BoosterVersion):
    for x in data['rocket']:
       if x:
        # On ajoute le "/" avant l'ID x pour une URL valide
        response = requests.get("https://api.spacexdata.com/v4/rockets/" + str(x)).json()
        BoosterVersion.append(response.get('name'))




def getLaunchSite(data, Longitude, Latitude, LaunchSite):
    for x in data['launchpad']:
        if x:
            # Correction orthographe 'launchpads' et ajout du '/'
            response = requests.get("https://api.spacexdata.com/v4/launchpads/" + str(x)).json()
            Longitude.append(response.get('longitude'))
            Latitude.append(response.get('latitude'))
            LaunchSite.append(response.get('name'))

def getPayloadData(data, PayloadMass, Orbit):
    for load_list in data['payloads']:
            if load_list:
                load = load_list[0]
                # Ajout du '/' pour éviter le JSONDecodeError
                response = requests.get("https://api.spacexdata.com/v4/payloads/" + str(load)).json()
                PayloadMass.append(response.get('mass_kg'))
                Orbit.append(response.get('orbit'))

def getcoredata(data, Block, ReusedCount, Serial, Outcome, Flights, GridFins, Reused, Legs, LandingPad):
    for core_list in data['cores']:
        # Correction du TypeError : core_list est une liste, on prend le premier élément [0]
        core = core_list[0]
        if core.get('core') is not None:
            response = requests.get("https://api.spacexdata.com/v4/cores/" + str(core['core'])).json()
            Block.append(response.get('block'))
            ReusedCount.append(response.get('reuse_count'))
            Serial.append(response.get('serial'))
        else:
            Block.append(None)
            ReusedCount.append(None)
            Serial.append(None)
        
        # On utilise .get() pour sécuriser l'extraction
        Outcome.append(str(core.get('landing_success')) + ' ' + str(core.get('landing_type')))
        Flights.append(core.get('flight'))
        GridFins.append(core.get('gridfins'))
        Reused.append(core.get('reused'))
        Legs.append(core.get('legs'))
        LandingPad.append(core.get('landingpad'))