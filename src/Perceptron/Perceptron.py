import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Lendo os dados e criando o array com o nome das colunas.
data = pd.read_csv("data/mutuos_intercompany.csv")
yColumn = "default"

X = data.drop(yColumn, axis=1)
y = data[yColumn]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=12)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Perceptron())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

print(pipeline.named_steps["model"].coef_)
print(pipeline.named_steps["model"].intercept_)