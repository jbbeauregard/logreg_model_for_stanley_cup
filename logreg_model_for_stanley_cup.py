import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_excel('nhl_data\\nhl_18_last_seasons_data.xlsx') 

X = data.drop(['StanleyCup', 'Season', 'Team'], axis=1)
y = data['StanleyCup']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Model accuracy :", accuracy)

print(classification_report(y_test, predictions))

joblib.dump(model, 'logreg_model_for_stanley_cup.pkl')