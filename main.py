import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 8640 time step
# 5 minute interval
# total 43200 minutes
# 720 hours 
# 30 days

# dataset starts from 12 AM april 1st to 11.55 PM of 30 April

global URBAN_CORE_CSV, ADJ_URBAN_CORE_CSV, URBAN_MIX_CSV, ADJ_URBAN_MIX_CSV
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

def linear_regression(dataset_name): 
    # read speed matrix data
    df = pd.read_csv(dataset_name, header=None) 

    # speed timestep start from column 8
    # omit column 1 - 7 
    df = df.drop(df.columns[0:7], axis=1).reset_index(drop=True)

    # Check if there are any zero values in the entire DataFrame
    if (df == 0).any().any():
        print("There are zero values in the DataFrame.")
    else:
        print("There are no zero values in the DataFrame.")

    # Calculate the average of each column
    column_averages = df.mean().to_frame('Mean')

    # add lag
    column_averages['Lag_1'] = column_averages['Mean'].shift(12)

    X = column_averages.loc[:, ['Lag_1']].reset_index(drop=True)
    X.dropna(inplace=True)  # drop missing values in the feature set
    Y = column_averages.loc[:, 'Mean'].reset_index(drop=True)  # create the target
    Y, X = Y.align(X, join='inner')  # drop corresponding values in target

    num_train_rows = 6048 # 21 days
    # Select the rows for the training independent variable set
    X_train = X.iloc[:num_train_rows].reset_index(drop=True)

    # Select the rows for the test independent variable set (remaining rows)
    X_test = X.iloc[num_train_rows:].reset_index(drop=True)

    # Select the rows for the training dependent variable set
    Y_train = Y.iloc[:num_train_rows].reset_index(drop=True)

    # Select the rows for the test dependent variable set (remaining rows)
    Y_test = Y.iloc[num_train_rows:].reset_index(drop=True)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, Y_train)

    # Make predictions on the test data
    Y_pred = pd.Series(model.predict(X_test), index= X_test.index)

    # Evaluate the model's performance
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    # Print model information
    print("Linear Regression Model Information:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)

    # Plot original dataset
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(column_averages.index, column_averages['Mean'], linewidth=1.0, color='blue', label='Data value')  
    ax1.plot(column_averages.index, column_averages['Lag_1'], linewidth=1.0, color='red', label='Previous time step')  
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Speed")
    ax1.legend()

  
    ax2.plot(X_test.index, X_test['Lag_1'], label='test data',  linewidth=1.0, color='blue')  
    ax2.plot(Y_pred.index, Y_pred, label = 'prediction data',  linewidth=1.0, color='red')  
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Speed")
    ax2.legend()
    plt.savefig('Prediction & Test Dataset Comparison', bbox_inches='tight', dpi=300)

    plt.show()


if __name__ == "__main__":
    display_csv_size()
    linear_regression(URBAN_CORE_CSV)
    linear_regression(URBAN_MIX_CSV)
