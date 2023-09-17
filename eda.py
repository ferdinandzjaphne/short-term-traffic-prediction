import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 8647 time step
# 5 minute interval
# total 43235 minutes
# 720.583333333 hours 
# 30.0243055555 days

URBAN_CORE_CSV = 'urban-core.csv'
ADJ_URBAN_CORE_CSV = 'Adj(urban-core).csv'
URBAN_MIX_CSV = 'urban-mix.csv'
ADJ_URBAN_MIX_CSV = 'Adj(urban-mix).csv'

def eda(file_name):
    df = pd.read_csv(URBAN_CORE_CSV, header=None) 

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

    axes[2].boxplot(column_averages)
    axes[2].legend()

    plt.savefig('eda.png')
    plt.show()


eda(URBAN_CORE_CSV)
eda(URBAN_MIX_CSV)

