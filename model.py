# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Simple dataset of three four features
# First of all load csv dataset file into dataframe
dataset = pd.read_csv('hiring.csv')

# Print the dataset

# Feature engineering to replace each Null cell
dataset['experience'].fillna(0, inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# Now specify the input features for training
X = dataset.iloc[:, :-1]

# Now some features are string/text, we need to convert them to number
# Converting words to integer values
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'zero': 0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = np.array(dataset.iloc[:, -1]).reshape(-1, 1)

# Splitting training and test set
# Since we have a very small dataset, we will train our model with all availabe data

# Train your model using linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Fitting model with training data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

# Using predict() function to test your model
print("Test one prediction")
input_data = np.array([9, 8, 3]).reshape((1, -1))
print(model.predict(input_data))
