import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
    ("svc", SVC(random_state=RANDOM_STATE))
])

# configurar um espaço de busca
params_grid = {
    'svc__kernel': ['linear'],
    'svc__C': np.random.uniform(52, 87.5, 200),
    'svc__class_weight': [None]
}

# configurar um amostrador estratificado
splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

# configurar o nosso "experimentador"
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    scoring='precision',
    cv=splitter,
    refit=True,
    verbose=10,
    error_score=0
)
     
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

# analisar o desempenho final do melhor modelo
yhat_train = grid_search.best_estimator_.predict(X_train)
yhat_test = grid_search.best_estimator_.predict(X_test)

print('Desempenho - Base de Treino')
print(classification_report(y_train, yhat_train))

print('Desempenho - Base de Teste')
print(classification_report(y_test, yhat_test))