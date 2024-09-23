import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
data_path = r"C:\Users\tragu\OneDrive\Documents\New folder\Dataset .csv"
data = pd.read_csv(data_path)

# Handle missing values
data.fillna({'Cuisines': 'Unknown', 'Price range': 0, 'Aggregate rating': 0, 'Votes': 0}, inplace=True)

# Preprocess dataset
# Preprocess dataset
data['Cuisines'] = data['Cuisines'].fillna('').apply(lambda x: ','.join(sorted(x.split(','))))
data['Price range'] = data['Price range'].fillna(0).astype(int)  # Convert to integers
data['features'] = data['Cuisines'] + ' ' + data['Price range'].astype(str) + ' ' + data['Rating text'] + ' ' + data['Votes'].astype(str)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['features'])

# Function to recommend restaurants
def recommend_restaurants(user_preferences, top_n=5):
    # Create user profile
    user_profile = ' '.join([f"{k} {v}" for k, v in user_preferences.items()])
    user_profile_vector = tfidf_vectorizer.transform([user_profile])

    # Calculate cosine similarity
    cosine_similarities = linear_kernel(user_profile_vector, tfidf_matrix).flatten()

    # Get top recommended restaurants
    top_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    recommended_restaurants = data.iloc[top_indices]
    return recommended_restaurants[['Restaurant ID', 'Cuisines', 'Price range', 'Rating text', 'Votes']]

# Function to get user preferences
# Function to get user preferences
def get_user_preferences():
    cuisines = input("Enter preferred cuisine(s): ")
    price_range = int(input("Enter preferred price range (1-5): "))  # Convert input to integer
    rating_text = input("Enter preferred rating (e.g., Good, Very Good): ")
    return {'Cuisines': cuisines, 'Price range': price_range, 'rating text': rating_text}


# Get user preferences
user_preferences = get_user_preferences()

# Get restaurant recommendations
recommendations = recommend_restaurants(user_preferences)
print(recommendations)

