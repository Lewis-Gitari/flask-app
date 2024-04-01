import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

#read the data using pandas
df = pd.read_csv("iris.csv")

#split the data into dependent and independent variables
X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = df['variety']

#separate the data into train and test set
X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=50)

#scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#instantiate a classifier model
classifier = RandomForestClassifier()

#fit the model onto our data
classifier.fit(X_train, y_train)

#create a pickle file
pickle.dump(classifier, open("iris.pkl", "wb"))
