import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load dataset
data_path = r"C:\Users\tragu\OneDrive\Documents\New folder\Dataset .csv"  # Provide the path to your dataset
data = pd.read_csv(data_path)

# Step 1: Preprocess the dataset
# Handle missing values
data.fillna('Unknown', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['Cuisines', 'Rating text']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Step 2: Split data into features and target variable
X = data[['Cuisines', 'Price range', 'Rating text']]
y = data['Cuisines']  # Replace 'target_variable' with your actual target variable name

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a classification model for cuisine prediction
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Evaluate the model's performance
y_pred = clf.predict(X_test)
# Check if test labels and predictions are empty
if len(y_test) == 0 or len(y_pred) == 0:
    print("Test labels or predictions are empty.")
else:
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)

    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # Print classification report if there are predictions
    if len(set(y_pred)) > 0:
        print(classification_report(y_test, y_pred))
    else:
        print("No predictions made.")
