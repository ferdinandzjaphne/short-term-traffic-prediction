
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from main import URBAN_MIX_CSV, URBAN_CORE_CSV, ADJ_URBAN_CORE_CSV, ADJ_URBAN_MIX_CSV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Sample traffic data (replace with your own dataset)
# This is a simplified example with a single feature (traffic volume)
df = pd.read_csv(URBAN_CORE_CSV, header=None) 
df = df.drop(df.columns[0:7], axis=1).reset_index(drop=True)
df = df.iloc[:, 0:100]
    
# data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
data = df.to_numpy().reshape(-1, 1)

# Sample ODC matrix (replace with your own data)
# The ODC matrix is typically a 2D matrix where rows represent origins, and columns represent destinations.
# odc_matrix = np.array([[0, 10, 5], [15, 0, 8], [7, 12, 0]])

# Data preprocessing
# You can concatenate the ODC matrix with the traffic data
# data = np.concatenate((data, odc_matrix), axis=1)

# Data preprocessing
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

X, y = [], []
sequence_length = 100  # Adjust for your specific use case

for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Training Loss: {train_loss:.4f}')
print(f'Testing Loss: {test_loss:.4f}')

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Inverse transform to get original scale
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

# Visualize the results
plt.plot(data, label='Actual Data', marker='o')
plt.plot(range(sequence_length, sequence_length + len(y_train_pred)), y_train_pred, label='Training Predictions', marker='x')
plt.plot(range(sequence_length + len(y_train_pred), sequence_length + len(data)), y_test_pred, label='Testing Predictions', marker='x')
plt.xlabel('Time Steps')
plt.ylabel('Traffic Volume')
plt.legend()
plt.show()