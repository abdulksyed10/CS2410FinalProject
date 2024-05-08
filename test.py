# Step 1: Selecting Features
import pandas as pd

# Load the original dataset into a DataFrame
original_df = pd.read_csv('nbastats.csv')

# Select the desired columns
selected_columns = ['Season', 'Team', 'Championship', 'G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
                    '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%']

# Create a new DataFrame with the selected columns
new_df = original_df[selected_columns].copy()

new_df.fillna(new_df.mean(numeric_only=True), inplace=True)


# Step 2: Data Preprocessing
# Handle missing values, encode categorical variables, etc.

# Step 3: Splitting Data into Training and Validation Sets
from sklearn.model_selection import train_test_split

# Drop rows with missing values
new_df.dropna(inplace=True)

# Define features (X) and target variable (y) after dropping missing values
X = new_df.drop(['Season', 'Team', 'Championship'], axis=1)
y = new_df['Championship']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
model = RandomForestClassifier()

# Step 5: Model Training
# Train the model using the training data
model.fit(X_train, y_train)

# Step 6: Model Evaluation
from sklearn.metrics import accuracy_score, classification_report

# Predict on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Step 7: Predicting Champion for New Season
# Load the new season stats into a DataFrame
new_season_stats = pd.read_csv('test.csv')

# Extract the necessary features
new_season_features = new_season_stats[['G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
                                        '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%']]

# Make predictions using the trained model
predicted_champion = model.predict(new_season_features)

# Display the predicted champion
print("Predicted Champion for the New Season:", predicted_champion)