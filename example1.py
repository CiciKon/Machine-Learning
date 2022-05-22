import pandas as pd
import numpy as np
import matplotlib
from sklearn import datasets, svm
from statistics import mean
from statistics import median
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv (r'D:\DAI\πτυχιακη\cars\data.csv')
df.head()
df.dtypes
print (df)
print(df.columns)

#statistics
for index in range(16):
    if index == 2 or index == 4 or index == 5 or index == 8 or index == 12 or index == 13 or index == 14 or index == 15:
        sum_col = df.iloc[:,index]
        print ("Mean ", index, " : % s" % (np.mean(sum_col)))
        print ("Max ", index, " : % s" % (max(sum_col)))
        print ("Min ", index, " : % s" % (min(sum_col)))
        print ("Median ", index, " : % s" % (median(sum_col)))
        print ("Standard Deviation ", index, " : % s" % (np.std(sum_col)))

#Dropping irrelevant columns
df = df.drop(['Engine Fuel Type', 'Market Category', 'Vehicle Style', 'Popularity',
              'Number of Doors', 'Vehicle Size'], axis=1)
df.head()

df = df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type":
                        "Transmission", "Driven_Wheels": "Drive Mode","highway MPG": "MPG-H", "city mpg":
                        "MPG-C", "MSRP": "Price" })
df.head()

#Dropping the duplicate rows
df.shape
duplicate_rows_df = df[df.duplicated()]
print("number of dupicate rows:", duplicate_rows_df.shape)

df.count()
df = df.drop_duplicates()
df.head(5)
df.count()

#Dropping the missing or null values
print(df.isnull().sum())

df = df.dropna()    # Dropping the missing values.
df.count()
print(df.isnull().sum())   # After dropping the values

#Prediction Model Building
X = df[['Year', 'HP', 'Cylinders']]
y = df.Price

num = int(len(df)*0.8)
train = df[:num]
test = df[num:]
print ("Data:",len(df),", Train:",len(train),", Test:",len(test))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size = 0.2,
                                random_state = 8)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# creating an object of LinearRegression class
lr = LinearRegression()
# fitting the training data
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))

# make predictions
y_pred = lr.predict(X_test)
print(y_pred)

# evaluate predictions
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

print(cf_matrix)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#from sklearn.tree import DecisionTreeClassifier
#clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
#clf_en.fit(X_train, y_train)

#y_pred_en = clf_en.predict(X_test)


# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#from sklearn import tree
#tree.plot_tree(clf_en.fit(X_train, y_train))

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred_en)
#print('Confusion matrix\n\n', cm)

#regr = linear_model.LinearRegression()
#regr.fit(x_train,y_train)

#coefficients = regr.coef_
#intercept = regr.intercept_

#y_pred = regr.predict(x_test)
#print("Coefficients: \n", coefficients)
#print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
#print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

#np.set_printoptions(precision=2)




