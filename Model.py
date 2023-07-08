import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

Dataset = pd.read_csv("CarPrice_Assignment.csv")
Y_df = Dataset['price']
X_df = Dataset[['enginelocation', 'enginesize']].copy()
X_df['enginelocation'] = pd.factorize(X_df['enginelocation'])[0]
X_np = X_df.to_numpy()
Y_np = Y_df.to_numpy()
scaler = StandardScaler().fit(X_np)
X_st = scaler.transform(X_np)
x_train, x_test, y_train, y_test = train_test_split(X_st, Y_np)
Model = GradientBoostingRegressor(loss='huber', criterion='friedman_mse',
                                  n_estimators=100, learning_rate=0.1).fit(x_train, y_train)
print('Score правильных ответов на обучающей выборке ', Model.score(x_train, y_train))
print('Score правильных ответов на тестовой выборке ', Model.score(x_test, y_test))
print('MAPE правильных ответов на обучающей выборке ', mean_absolute_percentage_error(y_train, Model.predict(x_train)))
print('MAPE правильных ответов на тестовой выборке ', mean_absolute_percentage_error(y_test, Model.predict(x_test)))
print('Ожидаемое значение:', y_test[17], ', значение предсказанное моделью:', Model.predict([x_test[17]])[0])
