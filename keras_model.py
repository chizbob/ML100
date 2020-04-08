import pandas as pd
from keras.models import Sequential
from keras.layers import *

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop("total_earnings", axis=1).values
Y = training_data_df[["total_earnings"]].values

#define the model
model = Sequential()
model.add(Dense(50, input_dim=9, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss="mean_squared_error", optimizer="adam")

#train
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

#load separate test data set
test_data_df = pd.read_csv("sales_data_training_scaled.csv")

X_test = test_data_df.drop("total_earnings", axis=1).values
Y_test = test_data_df[["total_earnings"]].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("MSE for the test data set: {}".format(test_error_rate))

#load the data to make a prediction
X = pd.read_csv("proposed_new_product.csv").values

#make prediction
prediction = model.predict(X)

#grab the first element of the first prediction
prediction = prediction[0][0]

#rescale from the 0-1 range to dollar amount
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("revenue prediction: ${}".format(prediction))

#save the model
model.save("trained_model.h5")
print("model saved")
