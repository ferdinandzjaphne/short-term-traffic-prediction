import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 8647 time step
# 5 minute interval
# total 43235 minutes
# 720.583333333 hours 
# 30.0243055555 days

# dataset starts from 12 AM april 1st to 11.55 PM of 30 April

URBAN_CORE_CSV = 'urban-core.csv'
ADJ_URBAN_CORE_CSV = 'Adj(urban-core).csv'
URBAN_MIX_CSV = 'urban-mix.csv'
ADJ_URBAN_MIX_CSV = 'Adj(urban-mix).csv'

def data_highlight_graph():
    # read speed matrix data
    data_csv = pd.read_csv(URBAN_CORE_CSV, header=None) 
    adj_csv = pd.read_csv(ADJ_URBAN_CORE_CSV, header=None) 

    road_segments = data_csv.iloc[:, 0]

    # assign street segment name to adjacency matrix data 
    adj_csv.index = road_segments
    adj_csv.columns = road_segments
    
    # show graph of adjacency data 
    # Create a graph from the adjacency matrix
    G = nx.from_pandas_adjacency(adj_csv)

    # Draw and display the graph
    nx.draw(G, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')

    plt.title("Adj urban core graph")
    plt.show()
    plt.close()

def display_csv_size():
    csv_file_path = [URBAN_CORE_CSV, URBAN_MIX_CSV, ADJ_URBAN_CORE_CSV, ADJ_URBAN_MIX_CSV]
    for file_name in csv_file_path: 
        df = pd.read_csv(file_name, header=None)
        num_rows, num_columns = df.shape
        print(file_name, ' : ', num_rows, ' x ', num_columns)

def preprocess_dataset(): 
    # read speed matrix data
    data_core_csv = pd.read_csv(URBAN_CORE_CSV, header=None) 
    data_mix_csv = pd.read_csv(URBAN_MIX_CSV, header=None) 

    # speed timestep start from column 8
    # omit column 1 - 7 
    data_core_csv = data_core_csv.drop(data_core_csv.columns[0:7], axis=1)

    # take first row for example
    data_core_csv_train = data_core_csv.iloc[0]

    # Define the number of columns per row
    columns_per_row = 5
    num_columns = 8640

    # Calculate the number of resulting rows
    num_rows = num_columns // columns_per_row

    # Create a new DataFrame with the desired shape
    new_df = pd.DataFrame(np.reshape(data_core_csv_train.values, (num_rows, columns_per_row)))

    # Optionally, you can reset the index of the new DataFrame
    new_df.reset_index(drop=True, inplace=True)

    # Display the resulting DataFrame
    print(data_core_csv_train)
    print(new_df)

    



# LINEAR REGRESSION
def linear_regression():
    # read speed matrix data
    data_core_csv = pd.read_csv(URBAN_CORE_CSV, header=None) 
    data_mix_csv = pd.read_csv(URBAN_MIX_CSV, header=None) 

    last_column_index = -1
    Y = data_core_csv.iloc[:, last_column_index]
    
    # PREPROCESS dataset
    X_train, X_temp, Y_train, Y_temp = train_test_split(data_core_csv, Y, test_size=0.3, random_state=42)
    X_test, X_validation, Y_test, Y_validation = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Print the sizes of the sets
    print("Training set size:", len(X_train))
    print("Testing set size:", len(X_test))
    print("Validation set size:", len(X_validation))

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, Y_train)

    # Make predictions on the test data
    Y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    # Print model information
    print("Linear Regression Model Information:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)

    # Create subplots
    plt.figure(figsize=(12, 5))

    # Plot the predicted vs. actual values
    plt.subplot(1, 2, 1)
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")

    # Plot the original data points
    plt.subplot(1, 2, 2)
    print(len(Y))
    plt.scatter(data_core_csv, Y, label="Original Data", color="blue")

    # Plot the linear regression line
    plt.plot(data_core_csv, Y_pred, label="Linear Regression Line", color="red")

    # Add labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    plt.show()



display_csv_size()
# data_highlight_graph()
# test()
preprocess_dataset()
# linear_regression()
