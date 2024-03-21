import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the entire contents of the files
df_add = pd.read_excel("C:\\Users\\lenovo t490\\PycharmProjects\\reconciliation_AI\\files\\fake_transactions_acquirer_date_decale.xlsx")
df_m = pd.read_excel("C:\\Users\\lenovo t490\\PycharmProjects\\reconciliation_AI\\files\\fake_transactions_merchant.xlsx")

#check the files
print(df_add.info())
print(df_m.info())

# Keep only the 'Transaction_Date' column
df_add = df_add[['Transaction_Date']]
df_m = df_m[['Transaction_Date']]

#check the 2 columns
print(df_add.info())
print(df_m.info())

# Convert 'Transaction_Date' column to datetime objects with a specific format
df_add['Transaction_Date'] = pd.to_datetime(df_add['Transaction_Date'], format='%Y%m%d%H:%M:%S', errors="coerce").dt.strftime('%Y%m%d%H:%M:%S')
df_m['Transaction_Date'] = pd.to_datetime(df_m['Transaction_Date'], format='%Y%m%d%H:%M:%S', errors="coerce").dt.strftime('%Y%m%d%H:%M:%S')

# Calculate the difference between the dates in each row
df_add["diff"] = (pd.to_datetime(df_add["Transaction_Date"], format="%Y%m%d%H:%M:%S") - pd.to_datetime(df_m["Transaction_Date"], format="%Y%m%d%H:%M:%S")).dt.total_seconds() / 60
print(df_add['diff'])

# Calculate the mean of the 'diff' column
mean_diff = df_add['diff'].mean()

# Replace NaN values in the 'diff' column with the mean
df_add['diff'].fillna(mean_diff, inplace=True)

# Add labels based on your condition
df_add.loc[df_add['diff'] == 0, 'label'] = 'ok'
df_add.loc[df_add['diff'] > 60, 'label'] = 'not ok'
df_add.loc[(df_add['diff'] > 0) & (df_add['diff'] <= 60), 'label'] = 'possible'

# Use a model
# Initialize a Random Forest Classifier
clf = RandomForestClassifier()

# Assuming 'diff' is a list of feature columns and 'label' is the target column
# allocate 30% of your data to the test set and 70% to the training set
X_train, X_test, y_train, y_test = train_test_split(df_add[['diff']], df_add['label'], test_size=0.3)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# Drop date transaction column
df_add.drop(columns=['Transaction_Date'], inplace=True)

# Save the df_add dataframe with predictions to a new Excel file
df_add.to_excel("C:\\Users\\lenovo t490\\PycharmProjects\\reconciliation_AI\\files\\resultat_matching_fields.xlsx", index=False)
