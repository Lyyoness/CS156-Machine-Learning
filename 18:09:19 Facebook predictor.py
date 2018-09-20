import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

#import data
facebook = pd.read_csv('facebook_train.csv')
facebook_test = pd.read_csv('facebook_test.csv')

#split into training and testing -> change fb_y choice
#here if you want to predict something else
dummies_df = pd.get_dummies(facebook[['type']])
facebook = pd.concat([facebook,dummies_df], axis=1)

dummies_df_test = pd.get_dummies(facebook_test[['type']])
facebook_test = pd.concat([facebook_test,dummies_df_test], axis=1)
fb_x_test = facebook_test[['total_followers', 'category', 'month',
				'weekday', 'hour', 'paid', 'type_Link', 'type_Photo', 
				'type_Status']]
fb_y_test = facebook_test[['like']]

fb_x = facebook[['total_followers', 'category', 'month',
				'weekday', 'hour', 'paid', 'type_Link', 'type_Photo', 
				'type_Status']]
fb_y_comment = facebook[['comment']]
fb_y_like = facebook[['like']]
fb_y_share = facebook[['share']]
X_train, X_test, y_train, y_test = train_test_split(fb_x, fb_y_like, test_size=0.33, random_state=42)

#create and fit model
LR = LinearRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
y_pred_test = LR.predict(fb_x_test)

scores = cross_validate(LR, X_train, y_train, scoring='mean_squared_error', cv=5)
print(scores)


# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print("Mean squared error test: %.2f"
      % mean_squared_error(fb_y_test, y_pred_test))
# The coefficients
print('Coefficients: %0.2f', LR.coef_)


plt.scatter(fb_y_test, y_pred_test)
plt.show()
