import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Read the dataset
df = pd.read_csv('dataset/HeightWeight.csv')

# Convert height from inches to centimeters
df['Height(Inches)'] = df['Height(Inches)'] * 2.54

# Convert weight from pounds to kilograms
df['Weight(Pounds)'] = df['Weight(Pounds)'] * 0.45359237

# Rename the columns
df = df.rename(columns={'Height(Inches)': 'Height', 'Weight(Pounds)': 'Weight'})

# Get a quick look on data for model selection
plt.scatter(df['Weight'], df['Height'])
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Height vs. Weight')
plt.show()

X = df[['Height']]
y = df['Weight']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'fit_intercept': [True, False],
              'copy_X': [True, False],
              'n_jobs': [None, -1],
              'positive': [True, False]}

# Create the linear regression model
model = LinearRegression()

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5)
random_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Save the model
joblib.dump(best_model, "model.pkl")