import numpy as num
import csv
import sklearn.linear_model as ln
import matplotlib.pyplot as plt

# Multiple Linear Regression for LAB3, written by Çınar Gedizlioğlu

# Opening and reading from the .csv file.
with open("PDatac.csv") as f:
    csv_list = list(csv.reader(f))

x1_list = num.array([])
x2_list = num.array([])
x3_list = num.array([])
x4_list = num.array([])
x5_list = num.array([])
y_list = num.array([])
titles = []
titles_sorted = []

# Extract each column of the .csv file to a different numpy array except the first row
# Also extract the first row to the array called "titles"
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

X = num.vstack((ones, x1_list, x2_list, x3_list, x4_list, x5_list)).T
Y = y_list.T



coefficients = num.linalg.inv(num.dot(X.T, X))
coefficients = num.dot(coefficients, X.T)
coefficients = num.dot(coefficients, Y)


Y_hat = num.dot(X, coefficients)


temp_coef = coefficients
for i in range(len(coefficients)-1):
    titles_sorted.append(titles[temp_coef.argmax()-1])
    titles.remove(titles[temp_coef.argmax() - 1])
    temp_coef = num.delete(temp_coef, temp_coef.argmax())

print(titles_sorted)

plt.title("Residual Error Plot")
plt.scatter(Y_hat, Y_hat-Y)
plt.hlines(y=0, xmin=-2000, xmax=3000, linewidth=2)
plt.show()


