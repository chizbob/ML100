import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#loading data from CSV
training_data_df = pd.read_csv("sales_data_training.csv")
test_data_df = pd.read_csv("sales_data_test.csv")

#scaler
scaler = MinMaxScaler(feature_range=(0,1))

#scale both inputs and outputs
scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)

#to bring it back to the original values
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

#create a new scaled dataframe object
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

#save the scaled dataframe to new csv files
scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
scaled_training_df.to_csv("sales_data_test_scaled.csv", index=False)
