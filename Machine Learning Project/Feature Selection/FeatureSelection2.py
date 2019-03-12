from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('PDatac.csv', encoding='latin1')

X = dataset[['x1', 'x2', 'x3', 'x4', 'x5']]
y = dataset['Y']


regr = linear_model.LinearRegression()
regr.fit(X, y)

print('Intercept: \n', regr.intercept_)
# print('Coefficients: \n', regr.coef_)
print('x1: \n', regr.coef_[0])
print('x2: \n', regr.coef_[1])
print('x3: \n', regr.coef_[2])
print('x4: \n', regr.coef_[3])
print('x5: \n', regr.coef_[4])


