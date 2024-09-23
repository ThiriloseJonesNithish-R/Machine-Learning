import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data_path = r"C:\Users\tragu\OneDrive\Documents\New folder\Dataset .csv"
data = pd.read_csv(data_path)

# Define numeric and categorical features
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.drop('Aggregate rating')
categorical_features = data.select_dtypes(include=['object']).columns
# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Split the data into training and testing sets
X = data.drop(columns=['Aggregate rating'])  # Ensure 'Aggregate rating' is included
y = data['Aggregate rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define regression model
model = LinearRegression()

# Create a pipeline
regression_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('model', model)])

# Train the model
regression_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = regression_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
