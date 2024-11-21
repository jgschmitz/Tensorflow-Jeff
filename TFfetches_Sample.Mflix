import tensorflow as tf
import pymongo
from bson import ObjectId

# MongoDB connection string
CONNECTION_STRING = 'mongodb+srv://xxx:xxx@shared-demo.xhytd.mongodb.net/?retryWrites=true&w=majority'

# Connect to MongoDB
def connect_to_mongo(connection_string):
    client = pymongo.MongoClient(connection_string)
    db = client.sample_mflix
    return db.movies

# Fetch a single movie document
def get_sample_movie(movies_collection):
    movie = movies_collection.find_one()
    print("Movie Title:", movie.get('title', 'No Title'))

# TensorFlow - Simple Add Operation
def tensorflow_add():
    a = tf.constant(5)
    b = tf.constant(10)
    result = tf.add(a, b).numpy()
    print("Simple TensorFlow Add Result:", result)

# Define and train a simple TensorFlow model
def train_model(xs, ys):
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=10, verbose=0)  # Suppress training logs for simplicity
    return model

# Fetch and process movies with TensorFlow
def process_movies_with_tensorflow(movies_collection):
    movies = movies_collection.find({"year": {"$gt": 1990}}, {"title": 1, "imdb.rating": 1}).limit(10)
    for movie in movies:
        imdb_rating = movie.get('imdb', {}).get('rating', 0)
        if imdb_rating:
            doubled_rating = tf.multiply(tf.constant(imdb_rating), 2).numpy()
            print(f"Movie: {movie.get('title', 'Unknown')} - Original Rating: {imdb_rating}, Doubled Rating: {doubled_rating}")

# Main function
if __name__ == "__main__":
    # Connect to MongoDB
    movies_collection = connect_to_mongo(CONNECTION_STRING)

    # Fetch and display a sample movie
    get_sample_movie(movies_collection)

    # Simple TensorFlow operation
    tensorflow_add()

    # Train and use a simple model
    xs = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    ys = tf.constant([1.5, 3.0, 4.5, 6.0, 7.5])
    model = train_model(xs, ys)
    prediction = model.predict([6.0])[0][0]
    print(f"Prediction for input 6.0: {prediction:.2f}")

    # Process movies using TensorFlow
    process_movies_with_tensorflow(movies_collection)
