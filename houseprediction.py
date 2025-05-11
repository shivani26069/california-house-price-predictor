# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %%
df = pd.read_csv(r"C:\Users\Dell\Downloads\californiadataset\housing.csv")  # Replace with your actual file path


# %%

df.describe()

# %%
df.shape

# %%
df.isnull().values.any()

# %%
df.isnull().sum()

# %%

df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# %%
df

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()
sns.boxplot(x=df['median_house_value'])

# %% [markdown]
# there are outliers in the data check the boxplot

# %%
import numpy as np

df['total_rooms_log'] = np.log1p(df['total_rooms'])  # log1p handles 0s
df['total_bedrooms_log'] = np.log1p(df['total_bedrooms'])
df['population_log'] = np.log1p(df['population'])
df['households_log'] = np.log1p(df['households'])


# %%
# Only include numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()



# %%
# Countplot
sns.countplot(x='ocean_proximity', data=df)

# Boxplot to see impact on price
plt.figure(figsize=(8, 6))
sns.boxplot(x='ocean_proximity', y='median_house_value', data=df)


# %%

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Use it
df_no_outliers = remove_outliers_iqr(df, 'median_house_value')



# %%
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='median_house_value')
plt.title("Before removing outliers")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(data=df_no_outliers, x='median_house_value')
plt.title("After removing outliers")
plt.show()


# %%
df_no_outliers = df[df['median_house_value'] < 500001]


# %% [markdown]
# Traing the model using LinearRegression 

# %%
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load the dataset
df = pd.read_csv(r"C:\Users\Dell\Downloads\californiadataset\housing.csv")  # Replace with your actual file path

# 3. Handle missing values
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# 4. One-hot encode the 'ocean_proximity' column
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# 5. Define feature columns and target
features = [
    'longitude', 'latitude', 'housing_median_age', 
    'total_rooms', 'total_bedrooms', 'population', 
    'households', 'median_income', 
    'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 
    'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'
]
target = 'median_house_value'

# 6. Split into train and test sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# 8. Make predictions

# Make sure X_test is a DataFrame with column names
y_pred = model.predict(X_test)  # X_test is a DataFrame, no warning


# 9. Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# 10. Predict on new data (example input)
sample_input = np.array([[-122.2, 37.8, 30, 2000, 400, 900, 300, 5.0, 0, 0, 1, 0]])  # Adjust accordingly
predicted_price = model.predict(sample_input)
print(f"Predicted house price: ${predicted_price[0]:,.2f}")

# %%
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Initialize models
rf_model = RandomForestRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)

# Train models
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Predict
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

# Evaluate
def evaluate_model(y_true, y_pred, name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

evaluate_model(y_test, rf_preds, "Random Forest")
evaluate_model(y_test, xgb_preds, "XGBoost")


# %%
import numpy as np

# Example new data point (adjust values accordingly)
sample_input = np.array([[-119.5, 34.0, 25, 2500, 450, 1100, 350, 4.5, 1, 0, 0, 1]])  # Modify based on feature importance

# Reshape if needed (ensure it matches the model's expected format)
sample_input = sample_input.reshape(1, -1)

# Predict using the trained model
predicted_price = model.predict(sample_input)
print(f"Predicted house price: ${predicted_price[0]:,.2f}")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted - Linear Regression")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.show()


# %%
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)  # or xgb_model
plt.figure(figsize=(10, 6))
importances.sort_values().plot(kind='barh')
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.show()


# %%
move house Prediction model.ipynb r"C:\Users\Dell\OneDrive\Desktop\python files\califorinahousingmodel"


# %%



