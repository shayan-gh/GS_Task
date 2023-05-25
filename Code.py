# Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import csv
import jdatetime
from pandas.api.types import CategoricalDtype
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans

# load the dataset
df = pd.read_csv('.../sample_data.csv')

# PreProcessing
df['date'] = pd.to_datetime(df['date'])

def convert_to_jalali(date):
    jalali_date = jdatetime.date.fromgregorian(date=date)
    return jalali_date


## Apply the conversion function to create the 'jalali_date' column
df['jalali_date'] = df['date'].apply(convert_to_jalali)

## Dictionary mapping for Persian weekday names
persian_weekdays = {
    0: 'شنبه',
    1: 'یکشنبه',
    2: 'دوشنبه',
    3: 'سه شنبه',
    4: 'چهارشنبه',
    5: 'پنجشنبه',
    6: 'جمعه'
}

## Function to get the Persian weekday
def get_persian_weekday(date):
    persian_weekday = persian_weekdays[date.weekday()]
    return persian_weekday


## Apply the function to create the 'jalali_weekday' column
df['jalali_weekday'] = df['jalali_date'].apply(get_persian_weekday)


# REPORT 1

## Define the custom sort order for jalali_weekday
custom_sort_order = ['شنبه', 'یکشنبه', 'دوشنبه',
                     'سه شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه']

## Convert jalali_weekday to a Categorical data type with the custom sort order
weekday_cat_type = CategoricalDtype(categories=custom_sort_order, ordered=True)
df['jalali_weekday'] = df['jalali_weekday'].astype(weekday_cat_type)

## Group by 'jalali_weekday' and calculate mean and standard deviation of 'order_id'
report_1 = df.groupby('jalali_weekday')['order_id'].agg(['mean', 'std'])

## Sort the aggregated data by the custom sort order
report_1 = report_1.reindex(custom_sort_order)


# REPORT 2

## Filter the DataFrame for working days (شنبه to چهارشنبه)
working_days_data = df[df['jalali_weekday'].isin(
    ['شنبه', 'یکشنبه', 'دوشنبه', 'سه شنبه', 'چهارشنبه'])]

## Filter the DataFrame for non-working days (پنجشنبه and جمعه)
non_working_days_data = df[df['jalali_weekday'].isin(['پنجشنبه', 'جمعه'])]

## Calculate the daily demand on working days and non-working days
working_days_demand = working_days_data['jalali_weekday'].value_counts(
).sort_index()
non_working_days_demand = non_working_days_data['jalali_weekday'].value_counts(
).sort_index()

## Plot the histogram
plt.figure(figsize=(10, 6))
plt.bar(working_days_demand.index, working_days_demand.values,
        color='blue', label='Working Days')
plt.bar(non_working_days_demand.index, non_working_days_demand.values,
        color='red', label='Non-Working Days')

## Add labels to the bars
for days, counts in [(working_days_demand, 'blue'), (non_working_days_demand, 'red')]:
    for i, v in enumerate(days.values):
        plt.text(i, v, str(v), ha='center', va='bottom',
                 fontweight='bold', color='black')

## Add x-axis labels and tick labels
plt.xlabel('Jalali Weekday')
plt.ylabel('Demand')
plt.title('Distribution of daily demand')
plt.xticks(rotation=45, ha='right')

## Add legend
plt.legend()
plt.tight_layout()
plt.show()


# REPORT 3

## Calculate RFM values
rfm_data = df.groupby('user_id').agg({
    'date': lambda x: (pd.to_datetime('today') - pd.to_datetime(x.max())).days,
    'order_id': 'count',
    'total_purchase': 'sum'
}).rename(columns={
    'date': 'Recency',
    'order_id': 'Frequency',
    'total_purchase': 'Monetary'
})

## Normalize the RFM values
rfm_normalized = (rfm_data - rfm_data.mean()) / rfm_data.std()

## Function to perform K-means clustering and return cluster labels


def perform_clustering(k):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(rfm_normalized)
    cluster_labels = kmeans.labels_
    return cluster_labels


## Prompt user for the desired number of clusters (K)
k = int(input("Enter the desired number of clusters (K): "))
## Perform clustering and get cluster labels
cluster_labels = perform_clustering(k)

## Add cluster labels to rfm_data
rfm_data['Cluster'] = cluster_labels

## Calculate average RFM values for each cluster
cluster_averages = rfm_data.groupby('Cluster').mean()

## Display the average RFM values for each cluster
print("Average RFM values for {} clusters:".format(k))
print(cluster_averages)

## Scatter plot of Recency vs Frequency
plt.scatter(rfm_data['Recency'], rfm_data['Frequency'], c=rfm_data['Cluster'])
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Scatter Plot of Recency vs Frequency')
plt.show()