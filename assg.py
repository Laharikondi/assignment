


# Import necessary libraries
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("SynchrophasorAnalysis").getOrCreate()

# Data Preprocessing
def preprocess_data(df):
    # Handling missing values
    df = df.na.drop()  # Drop rows with missing values
    # Feature extraction
    df = df.withColumn("mean_voltage", mean("voltage").over())
    df = df.withColumn("stddev_voltage", stddev("voltage").over())
    return df

# Load dataset
df = spark.read.csv("path/to/synchrophasor_data.csv", header=True, inferSchema=True)
df = preprocess_data(df)

# Convert Spark DataFrame to Pandas for TensorFlow compatibility
pandas_df = df.toPandas()

# Data Preparation for LSTM Model
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 10
X, y = create_sequences(pandas_df['voltage'].values, sequence_length)
X = np.expand_dims(X, axis=2)

# Build LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=20, batch_size=32)

# Evaluate the model
loss = model.evaluate(X, y)
print(f'Model Loss: {loss:.4f}')

# Real-time Data Processing (Simplified Example)
def process_real_time_data(new_data):
    new_sequence = np.append(X[-1][1:], [new_data])
    new_sequence = np.expand_dims(new_sequence, axis=0)
    new_sequence = np.expand_dims(new_sequence, axis=2)
    prediction = model.predict(new_sequence)
    return prediction

# Example of processing a new data point
new_voltage = 1.234  # New synchrophasor voltage reading
predicted_value = process_real_time_data(new_voltage)
print(f'Predicted Voltage: {predicted_value[0][0]:.4f}')
