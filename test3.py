import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv('nbastats.csv')

# Feature selection
selected_features = ['G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'Championship']
data = data[selected_features]

data.info()

# Encode target variable
label_encoder = LabelEncoder()
data['Championship'] = label_encoder.fit_transform(data['Championship'])

# Split the data into train and test sets
X = data.drop('Championship', axis=1)
y = data['Championship']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train and evaluate RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train_imputed, y_train)
rf_y_pred = rf_model.predict(X_test_imputed)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred)

# Train and evaluate Gradient Boosting Classifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_imputed, y_train)
gb_y_pred = gb_model.predict(X_test_imputed)
gb_accuracy = accuracy_score(y_test, gb_y_pred)
gb_report = classification_report(y_test, gb_y_pred)

# Train and evaluate Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_imputed, y_train)
lr_y_pred = lr_model.predict(X_test_imputed)
lr_accuracy = accuracy_score(y_test, lr_y_pred)
lr_report = classification_report(y_test, lr_y_pred)

# Train and evaluate Support Vector Machine
svc_model = SVC()
svc_model.fit(X_train_imputed, y_train)
svc_y_pred = svc_model.predict(X_test_imputed)
svc_accuracy = accuracy_score(y_test, svc_y_pred)
svc_report = classification_report(y_test, svc_y_pred)

# Train and evaluate Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_imputed, y_train)
dt_y_pred = dt_model.predict(X_test_imputed)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_report = classification_report(y_test, dt_y_pred)

# Predicting champion for the new season
new_season_stats = pd.read_csv('test.csv')
new_season_features = new_season_stats[selected_features[:-1]]  # Exclude 'Championship' column

predicted_champion_rf = label_encoder.inverse_transform(rf_model.predict(new_season_features))
predicted_champion_gb = label_encoder.inverse_transform(gb_model.predict(new_season_features))
predicted_champion_lr = label_encoder.inverse_transform(lr_model.predict(new_season_features))
predicted_champion_svc = label_encoder.inverse_transform(svc_model.predict(new_season_features))
predicted_champion_dt = label_encoder.inverse_transform(dt_model.predict(new_season_features))

print("\nRandom Forest Classifier:")
print("Accuracy:", rf_accuracy)
print("Classification Report:\n", rf_report)
print("Predicted Champion for the New Season:", predicted_champion_rf)

print("\nGradient Boosting Classifier:")
print("Accuracy:", gb_accuracy)
print("Classification Report:\n", gb_report)
print("Predicted Champion for the New Season:", predicted_champion_gb)

print("\nLogistic Regression:")
print("Accuracy:", lr_accuracy)
print("Classification Report:\n", lr_report)
print("Predicted Champion for the New Season:", predicted_champion_lr)

print("\nSupport Vector Machine:")
print("Accuracy:", svc_accuracy)
print("Classification Report:\n", svc_report)
print("Predicted Champion for the New Season:", predicted_champion_svc)

print("\nDecision Tree Classifier:")
print("Accuracy:", dt_accuracy)
print("Classification Report:\n", dt_report)
print("Predicted Champion for the New Season:", predicted_champion_dt)
