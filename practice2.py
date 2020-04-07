import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data set from CSV file
training_data_df = pd.read_csv("/Users/sokim/Downloads/Ex_Files_TensorFlow/Exercise_Files/03/sales_data_training.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_training = training_data_df.drop("total_earnings", axis=1).values
Y_training = training_data_df[["total_earnings"]].values

test_data_df = pd.read_csv("/Users/sokim/Downloads/Ex_Files_TensorFlow/Exercise_Files/03/sales_data_test.csv", dtype=float)

X_testing = test_data_df.drop("total_earnings", axis=1).values
Y_testing = test_data_df[["total_earnings"]].values

X_scaler = MinMaxScaler(feature_range=(0,1))
Y_scaler = MinMaxScaler(feature_range=(0,1))

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

#define model parameter
learning_rate = 0.001
training_epochs = 100
display_step = 5

#define how many inputs and outputs in our neural network
number_of_inputs = 9
number_of_outputs = 1

#define how many neurons for each layer
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

#section one: define the layers
#input layer
with tf.variable_scope("input"):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

#layer 1
with tf.variable_scope("layer_1"):
    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

#layer 2
with tf.variable_scope("layer_2"):
    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

#layer 3
with tf.variable_scope("layer_3"):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

#output layers
with tf.variable_scope("output"):
    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

#section two: define the cost function
with tf.variable_scope("cost"):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

#section three: define the optimizer function
with tf.variable_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#create a summary operation to log the progress
with tf.variable_scope("logging"):
    tf.summary.scalar("current_cost", cost)
    tf.summary.histogram("predicted_value", prediction)
    summary = tf.summary.merge_all()

saver = tf.train.Saver()

#initialize a session for TensorFlow to operate
with tf.Session() as session:
    #global variable initializer to initialize all the variables and layers
    session.run(tf.global_variables_initializer())

    #create log file writers to record training progress
    #store training and testing log data separately
    training_writer = tf.summary.FileWriter("./logs/training", session.graph)
    testing_writer = tf.summary.FileWriter("./logs/testing", session.graph)

    #run the optimizer over and over to train
    #one epoch is on full run
    for epoch in range(training_epochs):
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
        #every five training steps, log our progress
        if epoch % 5 == 0:
            #get the current accuracy scores
            training_cost, training_summary = session.run([cost, summary], feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

            #write the current training status to the log file
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})



    print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch, training_cost, testing_cost))

    save_path = saver.save(session, "logs/trained_model.ckpt")
    print("model saved: {}".format(save_path))
