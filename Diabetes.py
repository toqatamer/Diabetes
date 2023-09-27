import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import psutil
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('diabetes_prediction_dataset.csv')
df=pd.DataFrame(df)
(df.head(1))

#EDA
df['diabetes'].value_counts().plot(kind='bar')

sns.scatterplot(x='age',y='bmi',data=df,hue='gender')

sns.histplot(df['age'],kde=True,bins=10)

sns.scatterplot(x='age',y='bmi',data=df,hue='diabetes')

sns.displot(df['bmi'],kde=True,bins=15)

sns.barplot(x="smoking_history",y="diabetes",data=df)

sns.barplot(x="smoking_history",y="heart_disease",data=df)

sns.barplot(x="gender",y="diabetes",data=df)

sns.boxplot(x="gender",y="age",data=df,hue='diabetes')

sns.boxplot(x="diabetes",y="bmi",data=df)

plt.figure(figsize = [8, 8])
gen_percentage = df['heart_disease'].value_counts(normalize=True)
plt.pie(gen_percentage, labels = gen_percentage.index, counterclock = False,autopct='%1.0f%%')

sample = df.blood_glucose_level.sample(500)
sample.plot.hist(figsize=(15,10))

#data preprocessing
df.isna().sum()

df.duplicated().sum()

df=df.drop_duplicates()
df

df.info()

df.describe()

sns.pairplot(df)

#Correlation plot of the independent variables

sns.heatmap(df.corr(), annot = True, cmap = "coolwarm")

from sklearn.preprocessing import StandardScaler
#1.Create an instance of standard scaler class
scaler= StandardScaler()

df[['age','bmi']]= scaler.fit_transform(df[['age','bmi']])
print(df)

Encoded_df=pd.get_dummies(df['smoking_history'],dtype=int)
print(Encoded_df.columns)
df= pd.concat([df,Encoded_df],axis=1)
print(df.columns)
df= df.drop(['smoking_history'],axis=1)
print(df.columns)

from sklearn.preprocessing import StandardScaler
#1.Create an instance of standard scaler class
scaler= StandardScaler()
#%%
#2.fit and transform the data
df[['age','bmi']]= scaler.fit_transform(df[['age','bmi']])
print(df.head(5))

from sklearn.preprocessing import LabelEncoder
col =['gender']
df[col]=df[col].apply(LabelEncoder().fit_transform)
print(df.columns)
print(df.sample(3))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the data into a pandas DataFrame
data = pd.DataFrame({
    'gender': ['Female'],
    'age': [80.0],
    'hypertension': [0],
    'heart_disease': [1],
    'never': [1],
    'bmi': [25.19],
    'HbA1c_level': [6.6],
    'blood_glucose_level': [140],
    'diabetes': [0]
})

x= df.iloc[:, :7].join(df.iloc[:, -6:])
print(x.columns)

y= df.iloc[:,8]

print(y.head())


#Feature engineering:
1.using StandardScaler() to standardize the age and bmi
2.using encoding to change the smoking_history from categorical to numerical

we used classification as the output is categorical "diabetes"

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



from sklearn.metrics import classification_report
class_report - classification_report(y_test,knn_pred)
print(class_report)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
print(knn_pred)
print('KNN accuracy:', knn.score(x_test, y_test))

# Support Vector Machines (SVM)
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
print(svm_pred)
print('SVM accuracy:', svm.score(x_test, y_test))

# Decision Trees
tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)
tree_pred = tree.predict(x_test)
print(tree_pred)
print('Decision tree accuracy:', tree.score(x_test, y_test))

# Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(x_train, y_train)
forest_pred = forest.predict(x_test)
print(forest_pred)
print('Random forest accuracy:', forest.score(x_test, y_test))

# xgboost
from xgboost import XGBClassifier

xg_classifier = XGBClassifier(learning_rate = 0.5, n_estimators = 100)
xg_classifier.fit(x_train , y_train)
y_pred_xg = xg_classifier.predict(x_test)
print(y_pred_xg)
print('xgboost accuracy:',xg_classifier.score(x_test,y_test))

import numpy as np      # used for array
import pandas as  pd    # to get data frame to understand excel
import matplotlib as mp # to  draw chart
import seaborn as sns   # also to draw chart with addtion funictions
import sklearn as sk    # a comprehensive library that provides various tools

# Machine learning


#Implement Linear Regression model(LR)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
model1= linear_model.LinearRegression()

#Train the model by fitting  the training data to the created LR model
model1.fit(x_train,y_train)

#Prediction using test data
y_pred= model1.predict(x_test)
print(y_pred.shape)
print(y_pred)

print(model1.coef_)
print(model1.intercept_)

#Evaluate model performance
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))

#Check the LR model accuracy score:
from sklearn.metrics import accuracy_score
model1.score(x_test,y_test)

#Model evaluation
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAE = metrics.mean_absolute_error(y_test,y_pred)

MSE = metrics.mean_squared_error(y_test,y_pred)

RMSE = np.sqrt(MSE)

R2_score = metrics.r2_score(y_test,y_pred)

print(' mean_absolute_error= ', MAE)
print(' mean_squared_error= ', MSE)
print(' Root mean_squared_error= ', RMSE)
print(' R2_score= ', R2_score)
