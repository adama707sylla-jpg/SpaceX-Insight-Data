from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

SAMPLE_INPUT = {
    "FlightNumber": 85,          # numéro de vol SpaceX
    "PayloadMass": 5000,         # 5000 kg de charge utile
    "Reused": 1,                 # booster réutilisé (1=oui, 0=non)
    "Block": 5,                  # Block 5 = version la plus récente
    "Latitude": 28.5,            # Kennedy Space Center (Floride)
    "Longitude": -80.6,          # Kennedy Space Center (Floride)
    "Flight": 3,                 # 3ème vol de ce booster
    "Date": 2020,                # année de lancement
    "BoosterVersion": 5,         # Falcon 9 Block 5
    "Orbit": 3,                  # LEO = orbite basse terrestre
    "Serial": 42,                # numéro de série du booster
    "Outcome": 1,                # tentative d'atterrissage (1=oui)
    "LaunchSite": 1              # site de lancement encodé
}

# ── Test 1 : API vivante 
def test_root():
    r = client.get("/")
    assert r.status_code == 200

# ── Test 2 : Prédiction fonctionne 
def test_predict_status():
    r = client.post("/predict", json=SAMPLE_INPUT)
    assert r.status_code == 200

# ── Test 3 : Structure de la réponse 
def test_predict_structure():
    r = client.post("/predict", json=SAMPLE_INPUT)
    data = r.json()
    # Clés obligatoires dans TOUS les projets
    assert "prediction" in data
    assert "type" in data
    assert data["type"] in ["classification", "regression"]

# ── Test 4 : Logique selon le type 
def test_predict_logique():
    r = client.post("/predict", json=SAMPLE_INPUT)
    data = r.json()

    if data["type"] == "classification":
        # Vérifier que confiance est entre 0 et 1
        assert "confiance" in data
        assert 0.0 <= data["confiance"] <= 1.0
        # Vérifier que prediction est "0" ou "1"
        assert data["prediction"] in ["0", "1"]

    elif data["type"] == "regression":
        # Pas de confiance en régression
        assert "confiance" not in data
        # Vérifier que la valeur est un nombre positif
        assert isinstance(data["prediction"], (int, float))
        assert data["prediction"] > 0

# ── Test 5 : Mauvaise requête → erreur propre 
#def test_bad_request():
 #   r = client.post("/predict", json={})
  #  # Doit retourner 422 (données invalides)
   # assert r.status_code == 422