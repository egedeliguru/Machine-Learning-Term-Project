import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


dataset = pd.read_csv('PDatac.csv', encoding='latin1')

X = dataset[['x1', 'x2', 'x3', 'x4', 'x5']]
y = dataset['Y']

model = LogisticRegression()
rfe = RFE(model, 2)
fit = rfe.fit(X, y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("x1  x2  x3  x4  x5")
print("Feature Ranking: %s" % (fit.ranking_))