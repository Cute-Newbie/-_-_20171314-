import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import math 
from utils import*
import matplotlib.pyplot as plt

from models import Linear_Regression,Polynomial_Regression

# Train dataset 구성 
train = pd.read_csv('train.csv' )


x_train = train[['ID','Delivery_person_ID','Delivery_person_Age',
       'Delivery_person_Ratings', 'Restaurant_latitude',
       'Restaurant_longitude', 'Delivery_location_latitude',
       'Delivery_location_longitude', 'Type_of_order', 'Type_of_vehicle','Distance']]

y_train = train[['Time_taken(min)']] 


test = pd.read_csv('test.csv')
attr = ['Delivery_person_Age',
        'Delivery_person_Ratings',
        'Type_of_vehicle',
        'Distance']


x_test = test[['ID','Delivery_person_ID','Delivery_person_Age',
       'Delivery_person_Ratings', 'Restaurant_latitude',
       'Restaurant_longitude', 'Delivery_location_latitude',
       'Delivery_location_longitude', 'Type_of_order', 'Type_of_vehicle','Distance']]

y_test = test[['Time_taken(min)']]


model1 = Linear_Regression(x_train = x_train,
                          attr2use = attr,
                          y_train = y_train,
                          x_test = x_test)
model1.fit()
answer1 = model1.predict()

print("Linear_Regression")
print(f"MAE LOSS:{mae(answer1,y_test)}")
print(f"RMSE LOSS:{rmse(answer1,y_test)}")

model2 = Polynomial_Regression(
    degree = 2,
    x_train = x_train,
    attr2use = attr,
    y_train = y_train,
    x_test = x_test)
print("=======================================")
model2.fit()
answer2 = model2.predict()
print("Polynominal_Regression,degree=2")
print(f"MAE LOSS:{mae(answer2,y_test)}")
print(f"RMSE LOSS:{rmse(answer2,y_test)}")
print("=======================================")
model3 = Polynomial_Regression(
    degree = 3,
    x_train = x_train,
    attr2use = attr,
    y_train = y_train,
    x_test = x_test)
model3.fit()
answer3 = model3.predict()
#print(f"{model1.model_type}")
print("Polynominal Regression,degree=3")
print(f"MAE LOSS:{mae(answer3,y_test)}")
print(f"RMSE LOSS:{rmse(answer3,y_test)}")

print("=======================================")
model4 = Polynomial_Regression(
    degree = 4,
    x_train = x_train,
    attr2use = attr,
    y_train = y_train,
    x_test = x_test)
model4.fit()
answer4 = model4.predict()
#print(f"{model1.model_type}")
print("Polynominal Regression,degree=4")
print(f"MAE LOSS:{mae(answer4,y_test)}")
print(f"RMSE LOSS:{rmse(answer4,y_test)}")


print("=======================================")
model5 = Polynomial_Regression(
    degree = 5,
    x_train = x_train,
    attr2use = attr,
    y_train = y_train,
    x_test = x_test)
model5.fit()
answer5 = model5.predict()
#print(f"{model1.model_type}")
print("Polynominal Regression,degree=5")
print(f"MAE LOSS:{mae(answer5,y_test)}")
print(f"RMSE LOSS:{rmse(answer5,y_test)}")



