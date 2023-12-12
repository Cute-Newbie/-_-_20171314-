import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import math 


# 거리 고려
def haversine(dataframe):
    # 지구 반경 (km)
    R = 6371.0

    # 위도, 경도를 라디안으로 변환
    lat1_rad = math.radians(dataframe['Restaurant_latitude'])
    lon1_rad = math.radians(dataframe['Restaurant_longitude'])
    lat2_rad = math.radians(dataframe['Delivery_location_latitude'])
    lon2_rad = math.radians(dataframe['Delivery_location_longitude'])

    # 위도와 경도 차이
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # 허버사인 공식
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 거리 계산
    distance = R * c

    return distance



#MSE

def mse(prediction,answer):

    N = len(answer)

    if not isinstance(answer,np.ndarray):
        answer = answer.to_numpy()

    diff = np.sum((answer-prediction)**2)/N

    #print(diff)

    return diff



#RMSE
def rmse(prediction,answer):

    N = len(answer)

    if not isinstance(answer,np.ndarray):
        answer = answer.to_numpy()

    diff = np.sum((answer-prediction)**2)/N
    diff = math.sqrt(diff)
    #print(diff)

    return diff


def mae(prediction,answer):


    N = len(answer)

    if not isinstance(answer,np.ndarray):
        answer = answer.to_numpy()

    diff = np.sum(np.abs(answer-prediction))/N
    #diff = math.sqrt(diff)
    #print(diff)

    return diff




























