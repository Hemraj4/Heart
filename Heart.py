import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("heart.csv")
df.head()
df.columns
df.info()
df.describe()
df.duplicated().sum()
df['HeartDisease'].value_counts().plot(kind = "bar")
plt.show()

df.isnull().sum()
def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde = True)

plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)


plt.tight_layout()
plt.show()

df['Cholesterol'].value_counts()
ch_mean = df.loc[df['Cholesterol'] != 0,'Cholesterol'].mean()
df['Cholesterol'] = df['Cholesterol'].replace(0,ch_mean)
df['Cholesterol'] = df['Cholesterol'].round(2)
resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()

df['RestingBP'] = df['RestingBP'].replace(0, resting_bp_mean)

df['RestingBP'] = df['RestingBP'].round(2)
def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde = True)

plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)

plt.tight_layout()
plt.show()

sns.countplot(x = df['Sex'],hue = df['HeartDisease'])
plt.show()
sns.countplot(x = df['ChestPainType'],hue = df['HeartDisease'])
plt.show()
sns.countplot(x = df['FastingBS'],hue = df['HeartDisease'])
plt.show()
sns.boxplot(x = 'HeartDisease', y = 'Cholesterol',data = df)
plt.show()
sns.violinplot(x='HeartDisease', y='Age', data=df)
plt.show()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()


df_encode = pd.get_dummies(df,drop_first=True)
df_encode
df_encode = df_encode.astype(int)
df_encode


from sklearn.preprocessing import StandardScaler
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df_encode[numerical_cols] = scaler.fit_transform(df_encode[numerical_cols])
df_encode.head()
