import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from main import URBAN_MIX_CSV, URBAN_CORE_CSV, ADJ_URBAN_CORE_CSV, ADJ_URBAN_MIX_CSV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pickle


def mean_relative_error(y_true, y_pred):
    return K.abs((y_true - y_pred) / y_true)

def lstm_training(file_name, prediction_timestep, epoch, dataset_length, batch_size, plot_file_name):
    # Sample traffic data (replace with your own dataset)
    # This is a simplified example with a single feature (traffic volume)
    df = pd.read_csv(file_name, header=None) 
    df = df.drop(df.columns[0:7], axis=1).reset_index(drop=True)
    if dataset_length != 0:
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
    loss_functions = ['mean_absolute_error', 'mean_squared_error', mean_relative_error]

    # Create a list to store loss values for each function
    loss_values = []

    # Create lists to store training and testing loss for each function
    train_loss_values = []
    test_loss_values = []
    
    # Loop over the loss functions
    for loss_function in loss_functions:
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)

        # Train the model
        history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_test, y_test))

        # Evaluate the model
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print('Training Loss:',train_loss)
        print('Testing Loss:', test_loss)

        train_loss = history.history['loss'][-1]
        test_loss = history.history['val_loss'][-1]

        loss_values.append([train_loss, test_loss])

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
    ax1.set_title('Prediction Plot')
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)  # 2 rows, 2 columns, subplot 2
    print(loss_values)
    # Create a boxplot to compare the loss values
    boxplot = ax2.boxplot(loss_values, patch_artist=True)
    ax2.set_ylabel('Loss Value')
    ax2.set_title('Comparison of Train and Test Loss Values')

    # Set the labels for the box plot
    # ax2.set_xlabel('MAE', 'MSE', 'MRE')

    # Set colors for the box plots (optional)
    colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightcoral', 'lightcoral']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    # plt.show()
    plt.savefig(plot_file_name + ".png")

    pickle.dump(fig, open(plot_file_name + '.fig.pickle', 'wb')) # T

def show_plot():
    figx = pickle.load(open('core_15_mins_plot.fig.pickle', 'rb'))

    figx.show() # Show the figure, edit it, etc.!

if __name__ == "__main__":
    # 15 minutes
    lstm_training(URBAN_MIX_CSV, 3, 50, 0, 1000, "core_15_mins_plot")

    # 30 minutes
    lstm_training(URBAN_MIX_CSV, 6, 50, 0, 1000, "core_30_mins_mse_plot")

    # 60 minutes
    lstm_training(URBAN_MIX_CSV, 12, 50, 0, 1000, "core_60_mins_mse_plot")
