# 🚀 SpaceX Insight — Prédiction d'Atterrissage Falcon 9

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerisé-blue?style=flat-square&logo=docker)
![Accuracy](https://img.shields.io/badge/Accuracy-77.6%25-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-En%20production-brightgreen?style=flat-square)

> Pipeline Data Science complet pour prédire le succès d'atterrissage du booster Falcon 9 — de l'extraction via API REST jusqu'au déploiement cloud.

🌍 **API en production** → [spacex-insight-data.onrender.com](https://spacex-insight-data.onrender.com)  
📖 **Documentation** → [spacex-insight-data.onrender.com/docs](https://spacex-insight-data.onrender.com/docs)

---

## Résultats

| Métrique | Valeur |
|---|---|
| Algorithme | K-Nearest Neighbors (KNN) |
| Accuracy | **77.6%** |
| Type | Classification binaire (Succès / Échec) |
| Features | 13 paramètres de vol |

---

## Pipeline ML

```
Extraction données via API REST SpaceX
        ↓
Nettoyage + Feature Engineering
        ↓
Visualisations géospatiales (Folium + Dash)
        ↓
Tournoi de modèles → KNN sélectionné (77.6%)
        ↓
API FastAPI + Docker
        ↓
Déploiement Cloud (Render)
```

---

## Stack technique

| Catégorie | Outils |
|---|---|
| **Langage** | Python 3.11 |
| **Machine Learning** | Scikit-Learn, KNN, Feature Engineering |
| **Data** | Pandas, NumPy, API REST SpaceX |
| **Visualisation** | Folium, Dash, Matplotlib |
| **API** | FastAPI, Uvicorn |
| **Containerisation** | Docker |
| **Déploiement** | Render (Cloud) |

---

## Utilisation de l'API

### Tester en ligne
```bash
curl -X POST https://spacex-insight-data.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "FlightNumber": 100,
    "PayloadMass": 5000,
    "Reused": 1,
    "Block": 5,
    "Latitude": 28.5,
    "Longitude": -80.5,
    "Flight": 3,
    "Date": 2022,
    "BoosterVersion": 5,
    "Orbit": 1,
    "Serial": 200,
    "Outcome": 1,
    "LaunchSite": 1
  }'
```

### Réponse
```json
{
  "prediction": "1",
  "confiance": 1.0,
  "type": "classification"
}
```

- `prediction` : **1** = Atterrissage réussi / **0** = Échec
- `confiance` : probabilité entre 0 et 1
- `type` : classification

---

## Lancer en local

```bash
# Cloner le repo
git clone https://github.com/adama707sylla-jpg/SpaceX-Insight-Data.git
cd SpaceX-Insight-Data

# Avec Docker (recommandé)
docker build -t spacex-api .
docker run -p 8000:10000 spacex-api

# Sans Docker
pip install -r requirements.txt
uvicorn app:app --reload
```

Ouvre ensuite : `http://127.0.0.1:8000/docs`

---

## Structure du projet

```
SpaceX-Insight-Data/
├── app.py                  # API FastAPI universelle
├── mon_modele_knn.pkl      # Modèle KNN entraîné
├── mon_outillage.py        # Fonctions utilitaires
├── requirements.txt        # Librairies Python
├── Dockerfile              # Configuration Docker
└── README.md
```

---

## Points forts

- **Extraction automatisée** : Données récupérées via API REST SpaceX en temps réel
- **Visualisation géospatiale** : Carte interactive des sites de lancement avec Folium
- **Dashboard interactif** : Analyse des performances par orbite avec Dash
- **API Universelle** : Template réutilisable pour tout projet ML
- **Production ready** : Dockerisé et déployé sur le cloud

---

## Auteur

**Adama Sylla** — Étudiant Data Science (MIAGE L3)  
📧 Adama101sylla@gmail.com  
🌐 [Portfolio](http://adama707.pythonanywhere.com)  
💼 [GitHub](https://github.com/adama707sylla-jpg)

---

*Projet réalisé dans le cadre du IBM Data Science Professional Certificate*
