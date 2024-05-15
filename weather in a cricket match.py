#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dummy historical weather data (example features: temperature, humidity, wind speed)
# Assuming features are normalized between 0 and 1
X = np.array([[0.2, 0.5, 0.1],   # Example 1
              [0.8, 0.3, 0.7],   # Example 2
              [0.6, 0.9, 0.4],   # Example 3
              ...                # Add more examples
              ])
# Dummy labels indicating whether it rained (1) or not (0) during the cricket match
y = np.array([0, 1, 1, ...])  # Corresponding labels for the examples above

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Making predictions on the testing set
predictions = clf.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Now, you can use this trained model to predict whether it will rain during a cricket match
# by providing the weather conditions as input.

