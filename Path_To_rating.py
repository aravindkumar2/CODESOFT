
# pip install scikit-surprise

from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import GridSearchCV

# Load the dataset
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1, 5))
data = Dataset.load_from_file('path_to_ratings.csv', reader=reader)

# Split the dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use SVD algorithm
algo = SVD()

# Train the algorithm on the trainset
algo.fit(trainset)

# Make predictions on the test set
predictions = algo.test(testset)

# Evaluate predictions
accuracy.rmse(predictions)

# Make recommendations for a user (userID = 1)
user_id = 1
user_movies = [m for m, _ in trainset.ur[user_id]]
unseen_movies = [m for m in trainset.all_items() if m not in user_movies]

# Predict ratings for unseen movies
predictions = [algo.predict(user_id, movie) for movie in unseen_movies]

# Get top n recommendations
top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]

# Print top recommendations
for movie in top_n:
    print(movie.iid)
