import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data = pd.read_csv("data/mutuos_intercompany.csv")
yColumn = "default"
RANDOM_STATE = 12

X = data.drop(yColumn, axis=1)
y = data[yColumn]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

pipeline = Pipeline([
    ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))
])

params_grid = {
    'dt__criterion': ['entropy', 'log_loss'],
    'dt__max_depth': [9],
    'dt__splitter': ['best'],
    'dt__class_weight': [None],
    'dt__min_samples_split': range(3, 7)
}

splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

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