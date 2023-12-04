# THE AIM OF TASK

A movie recommendation system is a type of filtering system that predicts and suggests movies to users based on their preferences and viewing history. Collaborative filtering is a popular technique for building recommendation systems. It involves making automatic predictions about the preferences of a user by collecting preferences from many users (collaborating).

## PROBLEM STATEMENT

The goal is to develop a movie recommendation system using collaborative filtering and machine learning techniques in Python. The system should analyze user behavior, preferences, and interactions with movies to make personalized recommendations. The dataset used for training and testing the recommendation system should contain information about users, movies, and their interactions (e.g., ratings).

## ELEMENTS OF TASK

**COMMANDS USED:**

Building a movie recommendation system involves collaborative filtering, a technique that leverages user-item interactions to make predictions about a user's preferences. In this example, I'll go through building a collaborative filtering recommendation system using Python and the Surprise library, which is specifically designed for recommendation systems.

1. **Install Required Libraries**

         *pip install scikit-surprise*

2. **Import Libraries**  : Import necessary libraries

         *import pandas as pd*
         *from surprise import Dataset, Reader*
         *from surprise.model_selection import train_test_split*
         *from surprise import SVD, accuracy*
         *from surprise.dump import dump, load*

3. **Data Collection & Data Preprocessing** :

Obtain a dataset containing user ratings for movies. Handle missing data, if any.
Encode categorical variables.
Create a user-item matrix representing user ratings for each movie.

4. **Load and Explore Data**: Assuming you have a dataset with user-item interactions, where each row represents a user's rating for a particular movie.

         *Load your dataset*
         *df = pd.read_csv("movie_ratings_dataset.csv")*

         *Explore the dataset*
         *print(df.head())*

5. **Create a Surprise Dataset** : Surprise requires a specific format for the dataset, so create a Dataset object.

         *reader = Reader(rating_scale=(1, 5))*
         *data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)*

6. **Train-Test Split** : Split the dataset into training and testing sets.

         *trainset, testset = train_test_split(data, test_size=0.2, random_state=42)*

7. **Train a Collaborative Filtering Model** :
Use the Singular Value Decomposition (SVD) algorithm for collaborative filtering.

         *model = SVD(n_factors=100, random_state=42)model.fit(trainset)*

8. **Evaluate the Model** :

         *predictions = model.test(testset)accuracy.rmse(predictions)*

9. **Advanced Features Hyperparameter Tuning** :
Fine-tune the model's hyperparameters for better performance.

         *from surprise.model_selection import GridSearchCV*
         *param_grid = {'n_factors': [50, 100, 200], 'lr_all': [0.002, 0.005, 0.01], 'reg_all': [0.02 0.1, 0.2]}*
         *grid_search = GridSearchCV(SVD, param_grid, measures=['RMSE'], cv=5)*
         *grid_search.fit(data)*

         *best_model = grid_search.best_estimator['rmse']*
         *print(best_model)*

10. **Advanced Features & Cross-validation** :
Perform cross-validation to get a more robust evaluation.

         *from surprise.model_selection import cross_validate*

         *cross_validate(model, data, measures=['RMSE'], cv=5, verbose=True)*

11. **Save and Load the Model** : 
Save the trained model for future use.

         *dump("trained_model", algo=model)*
         *loaded_model = load("trained_model")*

12. **Make Recommendations** : 
Once your model is trained, you can use it to make movie recommendations for a specific user.

         *user_id = 1*
         *movie_ids = df['movie_id'].unique()*

 Predict ratings for movies the user hasn't seen

         *user_ratings = [(user_id, movie_id, 0) for movie_id in movie_ids if movie_id not in df[df['user_id'] == user_id]['movie_id'].values]*
         *predictions = model.test(user_ratings)*

Get top N recommendations

         *top_n = [(pred.iid, pred.est) for pred in predictions]*
         *top_n.sort(key=lambda x: x[1], reverse=True)*
         *top_n[:10]  # Display top 10 recommendations*

12. **Documentation and Reporting**:

Document the entire process, including data preprocessing, model selection, training, evaluation, and making recommendations.
Provide insights into the model's performance and limitations.








