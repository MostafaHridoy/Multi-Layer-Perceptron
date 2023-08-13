import seaborn as sns
import lime
import shap
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense 
#from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

df= pd.read_csv("I:\My_Project\healthcare-dataset-stroke-data.csv")

#print(df.head(5))

nan_values= df.isna()
nan_count = nan_values.sum()

print(nan_count)

'''plt.figure(figsize=(10, 6))
sns.heatmap(nan_values, cmap='viridis')
plt.title('NaN Values in Dataset')
plt.show()'''

df= df.dropna()

print("The number of nan values are : ",df.isna())

df=df.drop_duplicates()

print("The number of duplicated values are : ", df.duplicated().sum())

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['stroke']=label_encoder.fit_transform(df['stroke'])
df['smoking_status']=label_encoder.fit_transform(df['smoking_status'])
df['Residence_type']=label_encoder.fit_transform(df['Residence_type'])
df['work_type']=label_encoder.fit_transform(df['work_type'])
df['gender']=label_encoder.fit_transform(df['gender'])
df['ever_married']=label_encoder.fit_transform(df['ever_married'])

#print(df.head(5))

df= df.drop(['id'],axis=1)

print(df.head(2))

columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']

StandardScaler = StandardScaler()
columns_to_scale = ['age','avg_glucose_level','bmi']
df[columns_to_scale]= StandardScaler.fit_transform(df[columns_to_scale])

#print(df.head(5))


X=df.drop('stroke',axis=1)
Y=df['stroke']

X_train, X_test,y_train, y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

print('X_train-', X_train.size)
print('X_test-',X_test.size)
print('y_train-', y_train.size)
print('y_test-', y_test.size)

sm=SMOTE(random_state=42,sampling_strategy="auto",n_jobs=-1)
X_res,Y_res=sm.fit_resample(X_train,y_train)

print(X_res.size)
print(Y_res.size)

'''param_grid = {
    'hidden_layer_sizes': [(100, 100),(200,200),(300,300)],
    'activation': ['relu', 'tanh','sigmoid'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}'''

mlp_classifier=MLPClassifier(hidden_layer_sizes=(200,200),activation="relu",alpha=0.01, learning_rate="constant")

m= mlp_classifier.fit(X_res,Y_res)
p= m.predict(X_test)
c= confusion_matrix(y_test,p)
sns.heatmap(c,annot=True,cmap="rocket")
s=accuracy_score(y_test,p)
print("Accuracy Score of MLP : ",s)

print(classification_report(y_test,p))


