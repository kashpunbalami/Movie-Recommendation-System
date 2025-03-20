import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the models and data
@st.cache_data
def load_models():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('cosine_similarity.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    with open('movies.pkl', 'rb') as f:
        movie = pickle.load(f)
    with open('knn_model.pkl', 'rb') as f:
        knn1 = pickle.load(f)
    with open('movie_item_matrix.pkl', 'rb') as f:
        movie_item_matrix1 = pickle.load(f)
    return tfidf, cosine_sim, movie, knn1, movie_item_matrix1

# Load models and data
tfidf, cosine_sim, movie, knn1, movie_item_matrix1 = load_models()

# Function for Content-Based Filtering
def recommend_movies_by_genre(movie_title, num_recommendations=5):
    if movie_title not in movie['title'].values:
        return []

    movie_idx = movie[movie['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    return [movie.iloc[idx]['title'] for idx, score in sim_scores]

# Function for Collaborative Filtering (KNN)
def recommend_movies(movie_name, num_recommendations=5):
    if movie_name not in movie_item_matrix1.index:
        return []

    movie_idx = movie_item_matrix1.index.get_loc(movie_name)
    distances, indices = knn1.kneighbors(movie_item_matrix1.iloc[movie_idx].values.reshape(1, -1), n_neighbors=num_recommendations+1)

    return [movie_item_matrix1.index[i] for i in indices.flatten()[1:]]

# Hybrid Recommendation Function
def hybrid_recommendations(movie_name, cf_weight=0.5, cbf_weight=0.5, num_recommendations=5):
    cf_recs = recommend_movies(movie_name, num_recommendations)
    cbf_recs = recommend_movies_by_genre(movie_name, num_recommendations)

    hybrid_recs = {}
    for movie in cf_recs:
        hybrid_recs[movie] = hybrid_recs.get(movie, 0) + cf_weight
    for movie in cbf_recs:
        hybrid_recs[movie] = hybrid_recs.get(movie, 0) + cbf_weight

    hybrid_recs = sorted(hybrid_recs.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    return [movie for movie, score in hybrid_recs]

# Main App
def main():
    st.title("Movie Recommendation System")
    st.write("Welcome to the Movie Recommendation System! Select a recommendation model and a movie to get recommendations.")

    # Buttons for model selection
    model = st.radio(
        "Select a recommendation model:",
        ("Content-Based", "KNN", "Hybrid")
    )

    # Movie selection
    movie_list = movie['title'].tolist()
    selected_movie = st.selectbox("Choose a movie:", movie_list)

    # Get recommendations based on the selected model
    if st.button("Get Recommendations"):
        if model == "Content-Based":
            recommendations = recommend_movies_by_genre(selected_movie)
        elif model == "KNN":
            recommendations = recommend_movies(selected_movie)
        elif model == "Hybrid":
            recommendations = hybrid_recommendations(selected_movie)

        if recommendations:
            st.write("Recommended Movies:")
            for i, rec in enumerate(recommendations):
                st.write(f"{i+1}. {rec}")
        else:
            st.write("No recommendations found.")

if __name__ == "__main__":
    main()
