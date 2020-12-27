import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle



df = pd.read_csv('/home/vikas/Desktop/machine learning/machine_learning/logistic_reg_with_deplymnt/fire_data .csv')

#print(df.head(5))

# droping area column 

df = df.drop(['Area'], axis=1)
#print(df.head(5))

df = pd.DataFrame(df)

X =df.iloc[:, :3]
Y =df.iloc[:,-1]

x_train, x_test , y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 101)

fire_model = LogisticRegression()

fire_model.fit(x_train,y_train)


pickle.dump(fire_model, open('fire_model.pkl','wb'))
fire_model=pickle.load(open('fire_model.pkl', 'rb'))