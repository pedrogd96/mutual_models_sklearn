import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

data = pd.read_csv("data/mutuos_intercompany.csv")
yColumn = "default"
RANDOM_STATE = 12

X = data.drop(yColumn, axis=1)
y = data[yColumn]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="linear", class_weight=None, C=87.48635893714844, random_state=RANDOM_STATE))
])

pipeline.fit(X_train, y_train)

# Correios (tendência a inadimplência)
correios = [
    8000000000,# valor_contrato
    288,       # prazo_meses
    0.01531,   # taxa_juros
    0.04,      # ebitda_margem
    0.95,      # indice_liquidez_corrente
    0.92,      # indice_endividamento
    1.3,       # cobertura_juros
    0.35,      # volatilidade_receita
    -0.03,     # crescimento_receita_12m
    105,       # ciclo_caixa
    4,         # historico_atrasos_12m
    0.18,      # exposicao_cambial
    -0.07,     # variacao_capital_giro
    0.45,      # rating_interno_score
    0.35       # garantia_ratio
]

# Petrobras (tendência a adimplência)
petrobras = [
    3000000000,# valor_contrato
    84,        # prazo_meses
    0.0075,    # taxa_juros
    0.28,      # ebitda_margem
    1.85,      # indice_liquidez_corrente
    0.52,      # indice_endividamento
    5.6,       # cobertura_juros
    0.12,      # volatilidade_receita
    0.11,      # crescimento_receita_12m
    52,        # ciclo_caixa
    0,         # historico_atrasos_12m
    0.42,      # exposicao_cambial
    0.04,      # variacao_capital_giro
    0.91,      # rating_interno_score
    1.25       # garantia_ratio
]

# Array para teste nos modelos
X_empresas_df = pd.DataFrame([correios, petrobras], columns=X.columns)

pred = pipeline.predict(X_empresas_df)
print("Previsões:", pred)