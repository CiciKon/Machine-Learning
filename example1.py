import pandas as pd
from sklearn import datasets, svm
from statistics import mean
from statistics import median
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv (r'D:\DAI\πτυχιακη\cars\data.csv')
df.head()
print (df)
print(df.columns)

for index in range(16):
    if index == 2 or index == 4 or index == 5 or index == 8 or index == 12 or index == 13 or index == 14 or index == 15:
        sum_col = df.iloc[:,index]
        print ("Mean ", index, " : % s" % (np.mean(sum_col)))
        print ("Max ", index, " : % s" % (max(sum_col)))
        print ("Min ", index, " : % s" % (min(sum_col)))
        print ("Median ", index, " : % s" % (median(sum_col)))
        print ("Standard Deviation ", index, " : % s" % (np.std(sum_col)))

feature_cols = ['Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP',
               'Engine Cylinders', 'Transmission Type', 'Driven_Wheels',
               'Number of Doors', 'Vehicle Size', 'Vehicle Style',
               'highway MPG', 'city mpg', 'Popularity', 'MSRP']
X = df[feature_cols]
y = df.MSRP

num = int(len(df)*0.8)
train = df[:num]
test = df[num:]
print ("Data:",len(df),", Train:",len(train),", Test:",len(test))

#x_train = np.array(train[['Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP',
#       'Engine Cylinders', 'Transmission Type', 'Driven_Wheels',
#       'Number of Doors', 'Market Category', 'Vehicle Size', 'Vehicle Style',
#       'highway MPG', 'city mpg', 'Popularity']])
#y_train = np.array(train[['MSRP']])
#x_test = np.array(test[['Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP',
#      'Engine Cylinders', 'Transmission Type', 'Driven_Wheels',
#       'Number of Doors', 'Market Category', 'Vehicle Size', 'Vehicle Style',
#      'highway MPG', 'city mpg', 'Popularity']])
#y_test = np.array(test[['MSRP']])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, y_train)

y_pred_en = clf_en.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn import tree
tree.plot_tree(clf_en.fit(X_train, y_train))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_en)
print('Confusion matrix\n\n', cm)

#regr = linear_model.LinearRegression()
#regr.fit(x_train,y_train)

#coefficients = regr.coef_
#intercept = regr.intercept_

#y_pred = regr.predict(x_test)
#print("Coefficients: \n", coefficients)
#print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
#print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

#np.set_printoptions(precision=2)




