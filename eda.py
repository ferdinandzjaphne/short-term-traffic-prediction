import pandas as pd
import matplotlib.pyplot as plt
from main import URBAN_MIX_CSV, URBAN_CORE_CSV

# 8640 time step
# 5 minute interval
# total 43200 minutes
# 720 hours 
# 30 days

def eda(file_name):
    df = pd.read_csv(file_name, header=None) 

    max_speed = df.iloc[:, 7:].values.max()
    min_speed = df.iloc[:, 7:].values.min()
    avg = df.iloc[:, 7:].values.mean()
    print(file_name)
    print("max speed: ", max_speed)
    print("average speed: ", avg)
    print("min speed: ", min_speed)


    transposed_df = df.transpose()

    column_averages = transposed_df[7:].mean(axis=1).reset_index(drop=True)

    _, (axes) = plt.subplots(3, 1, sharex=False)
    # Plot raw dataset
    for column in transposed_df.columns[:7]:  # Exclude the X column
        axes[0].plot(transposed_df.index[7:], transposed_df[column][7:], label=transposed_df[column][0])
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Speed')
        axes[0].legend()

    axes[1].plot(column_averages.index, column_averages, label='Average Speed Among all Street Segment')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Speed')
    axes[1].legend()

    axes[2].boxplot(df.iloc[:3, 7:100])
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Speed')
    axes[2].legend()

    # Correlation Matrix
    # print(transposed_df[7:][:1])
    # correlation_matrix = transposed_df[7:][:1].corr(method='pearson')
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.9)

    plt.savefig('eda-pic.png')
    plt.show()

eda(URBAN_CORE_CSV)
eda(URBAN_MIX_CSV)