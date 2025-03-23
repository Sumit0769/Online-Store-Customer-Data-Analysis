#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

#Loading the dataset and seeing the information and description of the data
online_store_df=pd.read_csv('online_store_customer_data.csv')
print(online_store_df.info())
print(online_store_df.describe())

#Handling Missing Values to ensure data consistency and avoid calculation errors
online_store_df['Gender'].fillna('Unknown',inplace=True)
online_store_df['Employees_status'].fillna('Unknown',inplace=True)
online_store_df['Referal'].fillna(0, inplace=True)

#Checking what null age values to replaced with 
mean_age=online_store_df['Age'].mean()
median_age=online_store_df['Age'].median()
ax=online_store_df['Age'].plot(kind='hist',bins=10,color='skyblue',edgecolor='black')
ax.set_ylabel('FQ')
ax.set_xlabel('Age')
ax.axvline(mean_age,color='darkblue',linestyle='dashed',linewidth=2,label=f'Mean:{mean_age:.2f}')
ax.axvline(median_age,color='red',linestyle='dashed',linewidth=2,label=f'Median:{median_age:.2f}')
ax.legend()
plt.show()
print(np.mean(np.abs(online_store_df['Age']-mean_age)))
print(np.mean(np.abs(online_store_df['Age']-median_age)))
online_store_df['Age'].fillna(mean_age,inplace=True)



#Separating Date into Day, Month and Year which enables Time-based analysis and improves filtering and aggregation
online_store_df['Transaction_date']=pd.to_datetime(online_store_df['Transaction_date'])
online_store_df['day']=pd.DatetimeIndex(online_store_df.Transaction_date).day
online_store_df['month']=pd.DatetimeIndex(online_store_df.Transaction_date).month
online_store_df['year']=pd.DatetimeIndex(online_store_df.Transaction_date).year


#States with Highest Referral Rates
# Calculate total referrals by state
state_referrals = online_store_df.groupby('State_names')['Referal'].sum().sort_values(ascending=False)
print(state_referrals.head(10))  # Show top 10 states

#converts the Referal column from numerical values (0.0 and 1.0) into categorical values ('No' and 'Yes').
#This enhnaces data visualization and maintains consistency
online_store_df['Referal']=online_store_df['Referal'].astype(object)
online_store_df['Referal']=online_store_df['Referal'].replace({0.0:'No',1.0:'Yes'})

#Cleaning the dataset by removing rows where Amount_spent is missing and performs key analyses.
#This Ensures Accuracy and Avoids Errors in Computation 
data_cleaned = online_store_df.dropna(subset=['Amount_spent'])
missing_values_after_cleaning = data_cleaned.isnull().sum()

#Checking for missing values after cleaning, displaying dataset infotmation
#Ensures all NaN values in Amount_spent have been removed.
#Checks the total number of records left after cleaning. Ensures no missing values in critical columns.
print(missing_values_after_cleaning)
print(data_cleaned.info())

#Counting Payment Methods
#Identifies the most popular payment methods. Helps businesses understand customer payment preferences.
print(data_cleaned.Payment_method.value_counts())

#Identifying Top 10 States by customer count
#Helps identify where most customers are located. Useful for regional marketing strategies.
print(data_cleaned.State_names.value_counts().head(10))

#Checking customer segments. Helps understand which customer group is most active.
print(data_cleaned.Segment.value_counts())

#Calculating the average (mean) age of customers in each state.
#Identifies the average age of customers in different states. Helps businesses target specific age groups by location
St_age_df=data_cleaned.groupby('State_names')[['Age']].mean()
print(St_age_df)

#Segments Counts
print(online_store_df.Segment.value_counts())

#Spending Habits by Gender
# Calculate total and average spending by gender
gender_spending = data_cleaned.groupby('Gender')['Amount_spent'].agg(['sum', 'mean'])
print(gender_spending)

# Most Preferred Payment Method
# Calculate percentage distribution of each payment method
payment_distribution = data_cleaned['Payment_method'].value_counts(normalize=True) * 100
print(payment_distribution)

#States with Highest Spending
# Calculate total spending by state
state_spending = data_cleaned.groupby('State_names')['Amount_spent'].sum().sort_values(ascending=False)
print(state_spending.head(10))  # Show top 10 states

#Payment Method Preference by Segment
# Analyze payment methods by customer segments
segment_payment = data_cleaned.groupby('Segment')['Payment_method'].value_counts()
print(segment_payment)



#Relationship Between Spending and Referrals
# Compare average spending between referred and non-referred customers
referral_spending = data_cleaned.groupby('Referal')['Amount_spent'].sum()
print(referral_spending)

#Analyzing referral trends by state and identifies which states have the highest total customer transactions.
#Identifies states with the highest customer activity. Shows how referrals impact customer transactions.
#Helps businesses target high-traffic regions with promotions. Provides insights into how effective referrals are in each state.
referral_counts = data_cleaned.groupby(['State_names', 'Referal']).size().unstack(fill_value=0)
referral_counts.columns = ['No', 'Yes']
referral_counts['Total'] = referral_counts['Yes'] + referral_counts['No']
print(referral_counts)
print(referral_counts.Total.nlargest(10))

#Customizing the appearance of visualizations and sets up a multi-plot figure for displaying multiple charts in a grid format.
#Allows multiple plots to be displayed in one figure.
sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=10
matplotlib.rcParams['figure.figsize']=(12,8)
matplotlib.rcParams['figure.facecolor']='#00000000'
fig,axes=plt.subplots(4,2,figsize=(12,8))

#Percentage Of payment Methods
Pay_per=data_cleaned.Payment_method.value_counts()*100/data_cleaned.Payment_method.count()
#print(Pay_per)
axes[0,0].pie(Pay_per,labels=Pay_per.index,explode=(0.1,0.0,0.0),autopct='%1.1f%%')
axes[0,0].set_title('Payment Method Distribution')

#Amount Spent Month wise
monthly_g=data_cleaned.groupby('month').Amount_spent.sum()
sns.barplot(x=monthly_g.index,y=monthly_g,ax=axes[0,1],color='yellow')
axes[0,1].set_title('Monthly Spending Trends')
axes[0,1].set_xlabel('Month')
axes[0,1].set_ylabel('Total Amount Spent')

#Amount Spent State Wise
States_g=data_cleaned.groupby('State_names').Amount_spent.sum()
#print(States_g.nlargest(10))
sns.barplot(x=States_g.nlargest(10),y=States_g.nlargest(10).index,ax=axes[1,0],color='red')

#Number Of People
sns.histplot(x=online_store_df.Age,hue=online_store_df.Referal,bins=np.arange(10,80,10),ax=axes[1,1])
axes[1,1].set_xlabel('Age')
axes[1,1].set_ylabel('Number Of People')

#Amount Spent Year Wise
years_g=data_cleaned.groupby('year').Amount_spent.sum()
#print(years_g)
sns.barplot(y=years_g,x=years_g.index,ax=axes[2,0],color='blue')

#Distribution of Total Spending Among Customers
# Plot the distribution of total spending
sns.histplot(data_cleaned['Amount_spent'], bins=30,ax=axes[2,1])
axes[2,1].set_title('Distribution of Customer Spending')
axes[2,1].set_xlabel('Amount Spent')
axes[2,1].set_ylabel('Frequency')

time_series = data_cleaned.groupby('Transaction_date')['Amount_spent'].sum()
sns.lineplot(data=time_series, color='blue', marker='o',ax=axes[3,0])
axes[3,0].set_title('Total Amount Spent Over Time')
axes[3,0].set_xlabel('Date')
axes[3,0].set_ylabel('Amount Spent (in USD)')
axes[3,0].grid(True)

data_cleaned['YearMonth'] =data_cleaned['Transaction_date'].dt.to_period('M')

monthly_data = data_cleaned.groupby('YearMonth').agg(
    total_revenue=('Amount_spent', 'sum'),
    avg_revenue=('Amount_spent', 'mean'),
    transaction_count=('Transaction_ID', 'count')
).reset_index()
monthly_data['YearMonth'] = monthly_data['YearMonth'].dt.to_timestamp()
axes[3, 1].plot(monthly_data['YearMonth'], monthly_data['total_revenue'], color='b', marker='o', label='Total Revenue')
axes[3, 1].set_xlabel('Date')
axes[3, 1].set_ylabel('Total Revenue', color='b')
axes[3, 1].tick_params(axis='y', labelcolor='b')
ax2 = axes[3, 1].twinx()
ax2.plot(monthly_data['YearMonth'], monthly_data['transaction_count'], color='r', marker='s', label='Transaction Count')
ax2.set_ylabel('Transaction Count', color='r')
axes[3, 1].set_title('Business Growth Analysis - Monthly Revenue and Transaction Count')
axes[3, 1].legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout(pad=1)
plt.show()
