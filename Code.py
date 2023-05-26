import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import csv
import jdatetime
from pandas.api.types import CategoricalDtype
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
# load the dataset
df = pd.read_csv('C:/Users/TUF Gaming/Downloads/Documents/sample_data.csv')

df['date'] = pd.to_datetime(df['date'])


def convert_to_jalali(date):
    jalali_date = jdatetime.date.fromgregorian(date=date)
    return jalali_date


# Apply the conversion function to create the 'jalali_date' column
df['jalali_date'] = df['date'].apply(convert_to_jalali)

# Dictionary mapping for Persian weekday names
persian_weekdays = {
    0: 'shanbe',
    1: '1 shanbe',
    2: '2 shanbe',
    3: '3 shanbe',
    4: '4 shanbe',
    5: '5 shanbe',
    6: 'jome'
}

# Function to get the Persian weekday


def get_persian_weekday(date):
    persian_weekday = persian_weekdays[date.weekday()]
    return persian_weekday


# Apply the function to create the 'jalali_weekday' column
df['jalali_weekday'] = df['jalali_date'].apply(get_persian_weekday)


# REPORT 1

# Group by 'jalali_weekday' and calculate mean and standard deviation of count of the 'order_id'
daily_demand = df.groupby(['date', 'jalali_weekday'])[
    'order_id'].count().reset_index()
report_1 = daily_demand.groupby('jalali_weekday')[
    'order_id'].agg(['mean', 'std'])

# Define the custom sort order for jalali_weekday
custom_sort_order = ['shanbe', '1 shanbe', '2 shanbe',
                     '3 shanbe', '4 shanbe', '5 shanbe', 'jome']

# Sort the aggregated data by the custom sort order
report_1 = report_1.reindex(custom_sort_order)


# REPORT 2

# Filter the DataFrame for working days (shanbe to 4 shanbe)
working_days_data = df[df['jalali_weekday'].isin(
    ['shanbe', '1 shanbe', '2 shanbe', '3 shanbe', '4 shanbe'])]

# Filter the DataFrame for non-working days (5 shanbe and jome)
non_working_days_data = df[df['jalali_weekday'].isin(['5 shanbe', 'jome'])]

# Calculate the daily demand on working days and non-working days
working_days_demand = working_days_data['jalali_weekday'].value_counts(
).sort_index()
non_working_days_demand = non_working_days_data['jalali_weekday'].value_counts(
).sort_index()

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.bar(working_days_demand.index, working_days_demand.values,
        color='blue', label='Working Days')
plt.bar(non_working_days_demand.index, non_working_days_demand.values,
        color='red', label='Non-Working Days')

# Add labels to the bars
for days, counts in [(working_days_demand, 'blue'), (non_working_days_demand, 'red')]:
    for i, v in enumerate(days.values):
        plt.text(i, v, str(v), ha='center', va='bottom',
                 fontweight='bold', color='black')

# Add x-axis labels and tick labels
plt.xlabel('Jalali Weekday')
plt.ylabel('Demand')
plt.title('Distribution of daily demand')
plt.xticks(rotation=45, ha='right')

# Add legend
plt.legend()

plt.tight_layout()
plt.show()


# REPORT 3

# Calculate RFM values
rfm_data = df.groupby('user_id').agg({
    'date': lambda x: (pd.to_datetime('today') - pd.to_datetime(x.max())).days,
    'order_id': 'count',
    'total_purchase': 'sum'
}).rename(columns={
    'date': 'R',
    'order_id': 'F',
    'total_purchase': 'M'
})

standardize = StandardScaler()
standardize.fit(rfm_data)
rfm_data_T = standardize.fit_transform(rfm_data)

rfm_data_T = pd.DataFrame(rfm_data_T)
rfm_data_T.columns = ['R', 'F', 'M']

#n_clusters = int(input("Enter the desired number of clusters (K): "))
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters).fit(rfm_data_T)

pd.DataFrame(kmeans.cluster_centers_, columns=rfm_data_T.columns)

rfm_data_org = standardize.inverse_transform(rfm_data_T)
rfm_data_org = pd.DataFrame(rfm_data_org)
rfm_data_org.columns = ['R', 'F', 'M']
rfm_data_org['Clusters'] = kmeans.labels_
rfm_data_org.groupby('Clusters').agg('mean')
#rfm_data_org['UserId'] = rfm_data.index
#rfm_data_org['M'] = rfm_data_org['M'].astype(np.int64)
#labels = kmeans.labels_
#average_silhouette = silhouette_score(rfm_data_T, labels)
#sample_silhouette_values = silhouette_samples(rfm_data_T, labels)
# rfm_data_org.groupby(['Clusters']).size().reset_index(name='counts')

# Scatter plot of Recency vs Frequency
plt.scatter(rfm_data_org['R'], rfm_data_org['F'], c=rfm_data_org['Clusters'])
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Scatter Plot of Recency vs Frequency')
plt.show()
