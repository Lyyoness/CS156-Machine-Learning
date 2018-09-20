import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


#import data
casualty = pd.read_csv('casualty_train.csv')

#split into training and testing
#create dummy variables
dummies_df = pd.get_dummies(casualty[['casualty_class', 'gender',
				'travel']])
casualty = pd.concat([casualty,dummies_df], axis=1)

casualty_label = pd.get_dummies(casualty[['severe']])
casualty_feature = casualty.drop(['casualty_class', 'gender', 'severe', 'pedestrian_location',
					'pedestrian_movement', 'travel'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(casualty_feature, casualty_label, test_size=0.33, random_state=42)

#create and fit model
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

scores = cross_val_score(LR, X_train, y_train, cv=5)
print(np.mean(scores))

# The coefficients
print('Coefficients: %0.2f', LR.coef_)
