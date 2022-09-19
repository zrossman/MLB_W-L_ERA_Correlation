import kaggle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from kaggle.api.kaggle_api_extended import KaggleApi

#Downloading the dataset from Kaggle
api = KaggleApi()
api.authenticate()
api.dataset_download_file('omipelcastre/mlb-team-statistics', file_name = 'mlb_teams.csv')

#Extract ERA and W columns
df = pd.read_csv('mlb_teams.csv', index_col = 0)
df = df[['ERA', 'W']]

#Convert our feature and label columns into arrays
X = np.array(df['ERA'])
X = X.reshape(-1, 1)
y = np.array(df['W'])

#Splitting our data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

#Finding a line of best fit
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Using line of best fit on our test data
y_pred = regressor.predict(X_test)

#Comparing our predictions to the y_test data
comparison = []
for i in range(len(y_test)):
    a_list = []
    a_list.append(y_test[i])
    a_list.append(y_pred[i])
    comparison.append(a_list)

#Plotting our line of best fit over our train data
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('ERA vs Team Wins')
plt.xlabel('ERA')
plt.ylabel('Wins')
#plt.show()

accuracy = regressor.score(X_test, y_test)
print(accuracy)

#Our analysis shows that while there is a correlation between an MLB team's team ERA and wins, it is weak, and our
#accuracy mostly falls somewhere between 50% and 55%. This model was ran on data taken from season totals/averages for
#each team from 2012-2018. 