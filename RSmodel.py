import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('ratings.csv')
    df.columns = ['User Id', 'ProductId', 'Rating', 'Timestamp']
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').astype(int)
    df.drop(columns=['Timestamp'], inplace=True)

    # Filter users with at least 50 ratings
    top_users = df['User Id'].value_counts()
    df_final = df[df['User Id'].isin(top_users[top_users >= 50].index)]
    rating_matrix = df_final.pivot(index='User Id', columns='ProductId', values='Rating').fillna(0)

    return df_final, rating_matrix

df_final, rating_matrix = load_data()

# Find similar users
def similar_users(user_id, matrix):
    user_vector = matrix.loc[user_id].values.reshape(1, -1)
    similarity = cosine_similarity(matrix, user_vector).flatten()
    similar_indices = np.argsort(similarity)[::-1][1:11]  # Top 10 similar users
    return matrix.index[similar_indices].tolist()

# Get product recommendations
def get_recommendations(user_id, matrix):
    similar = similar_users(user_id, matrix)
    similar_users_ratings = matrix.loc[similar].mean(axis=0)
    user_ratings = matrix.loc[user_id]
    unseen_products = user_ratings[user_ratings == 0]
    recommendations = similar_users_ratings[unseen_products.index].sort_values(ascending=False)
    return recommendations.head(5)

# Streamlit UI
st.title("Product Recommendation System")

user_ids = rating_matrix.index.tolist()
user_input = st.selectbox("Select a User ID", user_ids)

if st.button("Get Recommendations"):
    recs = get_recommendations(user_input, rating_matrix)

    # Display only Product IDs
    product_ids = recs.index.tolist()
    st.write("### ✅ Top 5 Recommended Product IDs:")
    st.dataframe(pd.DataFrame(product_ids, columns=["Product ID"]))

    # Add bar chart visualization
    st.write("### Recommended Products Ratings:")
    st.bar_chart(recs)
