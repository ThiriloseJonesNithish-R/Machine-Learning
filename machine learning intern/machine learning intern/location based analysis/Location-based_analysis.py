import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap

# Load dataset
data_path = r"C:\Users\tragu\OneDrive\Documents\New folder\Dataset .csv"
data = pd.read_csv(data_path)

# Drop rows with missing latitude or longitude
data = data.dropna(subset=['Latitude', 'Longitude'])

# Explore latitude and longitude coordinates
plt.figure(figsize=(10, 8))
m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
m.drawcoastlines()
m.drawcountries()
m.scatter(data['Longitude'], data['Latitude'], latlon=True, s=50, alpha=0.5, zorder=2)
plt.title('Distribution of Restaurants')
plt.show()

# Group restaurants by city or locality
restaurant_count_by_city = data['City'].value_counts()

# Visualize restaurant distribution by city
plt.figure(figsize=(12, 6))
sns.countplot(x='City', data=data, order=data['City'].value_counts().index[:10])
plt.title('Top 10 Cities with Most Restaurants')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.show()

# Calculate average ratings by city
average_ratings_by_city = data.groupby('City')['aggregate_rating'].mean().sort_values(ascending=False)

# Visualize average ratings by city
plt.figure(figsize=(12, 6))
sns.barplot(x=average_ratings_by_city.index[:10], y=average_ratings_by_city.values[:10])
plt.title('Top 10 Cities with Highest Average Ratings')
plt.xlabel('City')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()

# Handle missing values in 'cuisines' column
data['cuisines'].fillna('Unknown', inplace=True)

# Calculate popular cuisines by city
popular_cuisines_by_city = data.groupby('City')['cuisines'].apply(lambda x: x.mode().iloc[0])

# Display popular cuisines by city
print("Popular Cuisines by City:")
print(popular_cuisines_by_city[:10])

# Calculate average price range by city
average_price_range_by_city = data.groupby('City')['Price range'].mean().sort_values(ascending=False)

# Visualize average price range by city
plt.figure(figsize=(12, 6))
sns.barplot(x=average_price_range_by_city.index[:10], y=average_price_range_by_city.values[:10])
plt.title('Top 10 Cities with Highest Average Price Range')
plt.xlabel('City')
plt.ylabel('Average Price Range')
plt.xticks(rotation=45)
plt.show()
