import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from main import URBAN_MIX_CSV, URBAN_CORE_CSV, ADJ_URBAN_CORE_CSV, ADJ_URBAN_MIX_CSV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

def lstm_training(file_name, prediction_timestep, loss_function, epoch, dataset_length, batch_size):
    # Sample traffic data (replace with your own dataset)
    # This is a simplified example with a single feature (traffic volume)
    df = pd.read_csv(URBAN_CORE_CSV, header=None) 
    df = df.drop(df.columns[0:7], axis=1).reset_index(drop=True)
    df = df.iloc[:, 0:dataset_length]
        
    data_raw = df.to_numpy().reshape(-1, 1)

    # Sample ODC matrix (replace with your own data)
    # The ODC matrix is typically a 2D matrix where rows represent origins, and columns represent destinations.
    # odc_matrix = np.array([[0, 10, 5], [15, 0, 8], [7, 12, 0]])

    # Data preprocessing
    # You can concatenate the ODC matrix with the traffic data
    # data = np.concatenate((data, odc_matrix), axis=1)

    # Data preprocessing
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data_raw)

    X, y = [], []
    sequence_length = prediction_timestep  # Adjust for your specific use case

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])

    X, y = np.array(X), np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    metrics=['accuracy']

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size)

    # Evaluate the model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print('Training Loss:',train_loss)
    print('Testing Loss:', test_loss)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Inverse transform to get original scale
    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_test_pred = scaler.inverse_transform(y_test_pred)

    # Visualize the results
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)  # 2 rows, 2 columns, subplot 1
    ax1.plot(range(0, len(data_raw)), data_raw, label='Actual Data', marker='o')
    ax1.plot(range(sequence_length, sequence_length + len(y_train_pred)), y_train_pred, label='Training Predictions', marker='x')
    ax1.plot(range(sequence_length + len(y_train_pred), sequence_length + len(y_train_pred) + len(y_test_pred)), y_test_pred, label='Testing Predictions', marker='x')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Traffic Speed')
    ax1.legend()
    # ax1.savefig("lstm-plot.png")

    ax2 = fig.add_subplot(2, 2, 2)  # 2 rows, 2 columns, subplot 2

    # Access the MSE values from the training history
    mse_values = history.history['loss']  # Change 'loss' to the appropriate metric name if needed
    print(mse_values)

    # Plot the MSE values
    epochs = range(1, len(mse_values) + 1)

    ax2.plot(epochs, mse_values, 'b', label='Training MSE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MSE')
    ax2.set_title('Training Mean Squared Error')
    ax2.legend()
    fig.savefig("lstm-plot.png")



# if __name__ == "__main__":
lstm_training(URBAN_CORE_CSV, 100, "mean_squared_error", 1, 100, 10)
