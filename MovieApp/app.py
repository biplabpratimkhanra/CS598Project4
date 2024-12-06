from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import math

app = Flask(__name__)

# Path for movie images
IMG_URL = "https://liangfgithub.github.io/MovieImages/"

# URL for movie data
url = "https://liangfgithub.github.io/MovieData/"

# Load movies data from the URL
moviesData = pd.read_csv(url + "movies.dat", sep="::", engine="python", encoding="ISO-8859-1", header=None)
moviesData.columns = ["MovieID", "Title", "Genres"]

# Convert MovieID in moviesData to match the format in the similarity matrix
moviesData['MovieID'] = moviesData['MovieID'].apply(lambda x: f"m{x}")

# Correctly format the poster image URL
moviesData['poster'] = IMG_URL + moviesData['MovieID'].str.replace('m', '') + '.jpg?raw=true'

# Load Rmat.csv locally
Rmat_path = "Rmat.csv"  # Ensure this path is correct and points to your local file
rating_matrix = pd.read_csv(Rmat_path, sep=",")

# Build the similarity matrix
def build_similarity_matrix_v2():
    print("Building Similarity Matrix")
    normalized_rating_matrix = rating_matrix.subtract(rating_matrix.mean(axis=1), axis='rows')
    cardinality_df = (~normalized_rating_matrix.isna()).astype('int')
    cardinality_df = cardinality_df.T
    cardinality_matrix = cardinality_df @ cardinality_df.T

    normalized_rating_matrix = normalized_rating_matrix.T.fillna(0)
    nr = normalized_rating_matrix @ normalized_rating_matrix.T
    squared_normalized_rating_matrix = ((normalized_rating_matrix**2) @ (normalized_rating_matrix != 0).T)
    squared_normalized_rating_matrix = squared_normalized_rating_matrix.apply(np.vectorize(np.sqrt))
    dr = squared_normalized_rating_matrix * squared_normalized_rating_matrix.T

    cosine_distance = nr / dr
    S = (1 + cosine_distance) / 2
    np.fill_diagonal(S.values, np.nan)
    S[cardinality_matrix < 3] = None
    return S

# Precompute similarity matrix
similarity_matrix = build_similarity_matrix_v2()

# Global variable to store user ratings
user_ratings = pd.Series(dtype=float)

# Recommendation function with zero division handling
def myIBCF(S, w, n=10):
    w = w.copy()
    identity = (~w.isna()).astype(int)
    w = w.fillna(0)
    
    #-w = w.reindex(S.columns, fill_value=0)
    #-identity = (~w.isna()).astype(int).reindex(S.columns, fill_value=0)
    S = S.copy().fillna(0)

    numerator = w.dot(S)
    denominator = identity.dot(S)

    # Prevent division by zero
    denominator = denominator.replace(0, np.nan)

    reco_movies = numerator / denominator
    print("reco_movies:", reco_movies)
    reco_movies = reco_movies.fillna(0)
    reco_movies = reco_movies.sort_values(ascending=False)[0:n].dropna()

    # Debugging: Print out the generated MovieID and Rating values
    print("Generated recommendations before conversion:")
    print(reco_movies)

  #  reco_moviesdf = pd.DataFrame({'MovieID': reco_movies.index, 'Rating': reco_movies.values})
    reco_moviesdf = pd.DataFrame({'MovieID': reco_movies.index, 'Rating': reco_movies.values})
    reco_moviesdf['Rating'] = reco_moviesdf['Rating'].apply(lambda x: round(x, 2))
    #math.ceil(num * factor) / factor

    # Debugging: Print out the MovieID column type and unique values
    print("MovieID type in recommendations:", reco_moviesdf['MovieID'].dtype)
    print("MovieID unique values in recommendations:", reco_moviesdf['MovieID'].unique())

    return reco_moviesdf



@app.route('/')
def home():
    movies = moviesData.head(100)
    movies = movies.to_dict(orient='records')
    return render_template('index.html', movies=movies)

@app.route('/rate', methods=['POST'])
def rate_movie():
    global user_ratings
    data = request.json
    print("Received ratings:", data)

    # Convert the received ratings to a pandas Series and ensure numeric types
    user_ratings = pd.Series(data).apply(pd.to_numeric, errors='coerce').reindex(similarity_matrix.columns, fill_value=0)

    # Debugging: Print the updated user ratings
    print("Updated user ratings:")
    print(user_ratings)

    return jsonify({"success": True, "message": "Ratings received!"})


@app.route('/recommend', methods=['GET'])
def recommend():
    # Ensure user ratings have been set
    if user_ratings.empty:
        return jsonify({"success": False, "message": "No user ratings available."})

    # Debugging: Print user ratings type and content
    print("User ratings type:", user_ratings.dtype)
    print("User ratings content:")
    print(user_ratings)

    # Generate recommendations using the updated user ratings
    recommendations = myIBCF(similarity_matrix, user_ratings, n=10)
    print("*********recommendations********:", recommendations)

    # Debugging: Print data types and unique values before merging
    print("Recommendations before merging:")
    print("MovieID type in recommendations:", recommendations['MovieID'].dtype)
    print("MovieID unique values in recommendations:", recommendations['MovieID'].unique())

    # Merge with movie data to get titles and posters
    recommendations = recommendations.merge(moviesData, on='MovieID', how='left')

    # Debugging: Print merged recommendations
    print("*********Recommendations after merging********:")
    print(recommendations)

    # Ensure the poster URL format is correct
    recommendations['poster'] = IMG_URL + recommendations['MovieID'].str.replace('m', '') + '.jpg?raw=true'

    result = recommendations[['Title', 'poster','Genres','Rating']].to_dict(orient='records')
    print("***************** result **************:", result)
    return jsonify(success=True, recommendations=result)


if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, port=5000)

