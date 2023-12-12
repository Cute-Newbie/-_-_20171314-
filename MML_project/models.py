import numpy as np
import pandas as pd 



#Linear Regression

class Linear_Regression:

    def __init__(self,x_train,attr2use,y_train,x_test):

        self.x_train = x_train[attr2use]
        

        self.x_test = x_test[attr2use]
        self.y_train = y_train
        self.weight = None
        self._name = "Linear Regressor"


    def fit(self):

        X = self.x_train.to_numpy()

        X = np.insert(X, 0, 1, axis=1)
        Y = self.y_train.to_numpy()


        # 회귀 계수 계산 (Beta = (X'X)^-1 X'Y)
        X_transpose = np.transpose(X)  # X'
        X_transpose_dot_X = X_transpose.dot(X)  # X'X
        inv_X_transpose_dot_X = np.linalg.inv(X_transpose_dot_X)  # (X'X)^-1
        X_transpose_dot_Y = X_transpose.dot(Y)  # X'Y
        beta = inv_X_transpose_dot_X.dot(X_transpose_dot_Y)  # Beta

        self.weight = beta
        
        return beta
    



    def predict(self):

        test = self.x_test.to_numpy()
        

        test_x = np.insert(test, 0, 1, axis=1)
      

        prediction = test_x.dot(self.weight)

        return prediction




#Polynominal Regression


class Polynomial_Regression:
    
    def __init__(self, degree,x_train,attr2use,y_train,x_test):
        self.degree = degree
        self.coefficients = None
        self.x_train = x_train[attr2use]
        

        self.x_test = x_test[attr2use]
        self.y_train = y_train
        self.weight = None
        self._name = "Linear Regressor"

    def _transform_input(self, X):
        """ 입력 데이터를 다항식으로 변환 """
        n_samples, n_features = X.shape
        X_poly = np.ones((n_samples, 1))

        for degree in range(1, self.degree + 1):
            for feature_index in range(n_features):
                X_poly = np.hstack((X_poly, X[:, feature_index:feature_index+1] ** degree))

        return X_poly

    def fit(self):
        """ 모델 학습 """
        X_poly = self._transform_input(self.x_train.to_numpy())

        # 계수 계산 (Beta = (X'X)^-1 X'Y)
        X_transpose = np.transpose(X_poly)
        X_transpose_dot_X = X_transpose.dot(X_poly)
        inv_X_transpose_dot_X = np.linalg.inv(X_transpose_dot_X)
        X_transpose_dot_Y = X_transpose.dot(self.y_train.to_numpy())
        self.coefficients = inv_X_transpose_dot_X.dot(X_transpose_dot_Y)

    def predict(self):
        """ 예측 수행 """
        X_poly = self._transform_input(self.x_test.to_numpy())
        return X_poly.dot(self.coefficients)

        return X_poly.dot(self.coefficients)


