import numpy as num
import csv
import math
import matplotlib.pyplot as plt

with open("PDatac.csv") as f:
    csv_list = list(csv.reader(f))

loocv_hat = num.array([])
x1_list = num.array([])
x2_list = num.array([])
x3_list = num.array([])
x4_list = num.array([])
x5_list = num.array([])
y_list = num.array([])
titles = []
titles_sorted = []

for row in csv_list:
    if row != csv_list[0]:
        x1_list = num.append(x1_list, int(row[1]))
        x2_list = num.append(x2_list, int(row[2]))
        x3_list = num.append(x3_list, float(row[3]))
        x4_list = num.append(x4_list, float(row[4]))
        x5_list = num.append(x5_list, float(row[5]))
        y_list = num.append(y_list, int(row[6]))
    else:
        titles.append(row[1])
        titles.append(row[2])
        titles.append(row[3])
        titles.append(row[4])
        titles.append(row[5])

ones = num.ones((1, len(x1_list)))
mse_loocv = 0

for i in range(len(x1_list)):  # Do the regression for each data point.

    # At each iteration, the i'th point becomes the test data and the rest becomes the train data.
    # We compute each prediction and store them.

    X = num.vstack((ones, x1_list, x2_list, x3_list, x4_list, x5_list)).T
    Y = y_list.T

    X_test = X[i]   # The i'th point becomes the test data.
    Y_test = Y[i]
    X_train = num.delete(X, i, 0)   # The rest becomes the train data.
    Y_train = num.delete(Y, i, 0)

    # Compute the coefficients just like we did in LAB3.
    coefficients = num.linalg.inv(num.dot(X_train.T, X_train))
    coefficients = num.dot(coefficients, X_train.T)
    coefficients = num.dot(coefficients, Y_train)

    Y_hat = num.dot(X_test, coefficients)   # Compute the prediction using the coefficients
    loocv_hat = num.append(loocv_hat, Y_hat)  # Append the prediction into the array "loocv_hat" for future plotting
    mse_loocv = mse_loocv + math.pow((Y_hat-Y_test), 2)   # Sum the squared errors

mse_loocv = mse_loocv / len(x1_list)   # Average the sum of squared errors

# Computing the regression one last time, as instructed in Task 2
coefficients = num.linalg.inv(num.dot(X.T, X))
coefficients = num.dot(coefficients, X.T)
coefficients = num.dot(coefficients, Y)

Y_hat = num.dot(X, coefficients)
mse_all = num.mean(num.square(Y_hat-Y))

print("MSE with loocv:" + str(mse_loocv))
print("MSE all:" + str(mse_all))

# Plotting the results
plt.title("Residual Error Plot")
plt.scatter(Y_hat, Y_hat-Y)
plt.scatter(loocv_hat, loocv_hat-Y)
plt.hlines(y=0, xmin=-2000, xmax=4000, linewidth=2)
plt.show()
