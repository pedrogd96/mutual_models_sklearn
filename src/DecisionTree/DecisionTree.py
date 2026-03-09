import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data = pd.read_csv("data/mutuos_intercompany.csv")
yColumn = "default"
RANDOM_STATE = 12

X = data.drop(yColumn, axis=1)
y = data[yColumn]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

pipeline = Pipeline([
    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
])

pipeline.fit(X_train, y_train)

yhat_train = pipeline.predict(X_train)
yhat_test = pipeline.predict(X_test)

print(classification_report(y_train, yhat_train))
print(classification_report(y_test, yhat_test))

# Analisando as dimensões da árvore criada
print(f'Profundidade da árvore: {pipeline.named_steps['model'].get_depth()}')
print(f'Número de folhas da árvore: {pipeline.named_steps['model'].get_n_leaves()}')