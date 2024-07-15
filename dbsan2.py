import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_2022_2023 = pd.read_csv("Combined Dataset (Jul'22 - Jun'23).csv")
data_2020_2021 = pd.read_csv("combined_datasets(2020-2021).csv")
data_2018_2019 = pd.read_csv("Combined_datasets(2018-2019).csv")

all_data = pd.concat([data_2022_2023, data_2020_2021, data_2018_2019], ignore_index=True)

monthly_stats = all_data.groupby(['Month']).agg({'ColumnO3': ['mean', 'median', 'std']})
monthly_stats.columns = ['Mean', 'Median', 'Deviation']


print("Monthly Statistics:")
print(monthly_stats)

overall_deviation = all_data.groupby(['Month'])['ColumnO3'].std()
print("\nDeviation Between All Months in All Years:")
print(overall_deviation)

plt.errorbar(monthly_stats.index, monthly_stats['Mean'], yerr=monthly_stats['Deviation'], fmt='o', capsize=5)
plt.title("Monthly Mean and Deviation")
plt.xlabel("Month")
plt.ylabel("ColumnO3 Mean")
plt.show()
