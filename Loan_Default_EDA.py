import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from matplotlib import cm

data_descriptions = pd.read_csv('data_descriptions.csv')
pd.set_option('display.max_colwidth', None)
print(data_descriptions)

train_df = pd.read_csv("train.csv")
print('train_df Shape:', train_df.shape)
print(train_df.head())

def frame_stats(df):
    stats = pd.DataFrame(columns=df.columns, index=('Type', 'Min', 'Max', 'Mean'))
    for col in df.columns:
        stats.loc['Type', col] = df[col].dtype
        if is_numeric_dtype(df[col].dtype):
            stats.loc['Min', col] = df[col].min()
            stats.loc['Max', col] = df[col].max()
            stats.loc['Mean', col] = df[col].mean()
    pd.set_option('display.max_rows', None)
    print(stats.transpose())
    print('Rows = ' + str(df.shape[0]))
    print('Cols = ' + str(df.shape[1]))
    
# Read dataframe and look at columns
frame_stats(train_df)
all_vars = []
for col in train_df.columns:
    all_vars.append(train_df[col].name)
cat_vars = all_vars[10:17]
num_vars = all_vars[1:10]
print(cat_vars)
print(num_vars)

# Categorical variables
cat_table = pd.DataFrame(cat_vars, columns=['name'])
cat_table['count'] = 0
for row in cat_table.index:
    vals = len(train_df[cat_table.iloc[row]['name']].unique())
    cat_table.iloc[row, 1] = vals
print(cat_table)

# First look at numerical variables 
train_df[num_vars].hist()
plt.show()

# Closer look at dependent variable
#fig = plt.figure()
#train_df['Default'].plot(kind='density')
#plt.show()

print(train_df.value_counts("Default"))

# Correlation matrix
nums_only = train_df[num_vars]
correlations = nums_only.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
fig.colorbar(cax)
ticks = np.arange(0,len(num_vars),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(num_vars)
ax.set_yticklabels(num_vars)
plt.show()

print(correlations)
