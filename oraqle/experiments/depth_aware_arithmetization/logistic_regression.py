from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the dataset
data = load_breast_cancer()
X = data.data  # type: ignore
y = data.target  # type: ignore

# Consider the range to be [-50, +50]
bound = 10_000
p = 786433
half = 10000 #p // 2

def fixed_prec(x: float) -> int:
    print(x)
    assert -bound < x < bound
    return round(x / bound * half) % p

X = np.vectorize(fixed_prec)(X)


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=10000, penalty=None, solver='newton-cg')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=data.target_names)  # type: ignore

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Get the coefficients and feature names
weights = model.coef_[0]
features = data.feature_names  # type: ignore

# Combine and sort by weight (optional)
feature_weights = list(zip(features, weights))
feature_weights.sort(key=lambda x: x[1], reverse=True)  # Sort by weight descending

# Print the weights
print("Feature Weights (sorted):")
for feature, weight in feature_weights:
    print(f"{feature}: {weight:.4f}")

print(model.intercept_)

# Get the weights and intercept
factor = 1
weights = np.round(model.coef_[0] * factor) % p       # Shape: (n_features,)
intercept = np.round(model.intercept_[0]) % p  # Scalar

# Compute the logits (z = w·x + b) for the test set
logits = np.dot(X_test, weights) + intercept

# Find the highest and lowest logits
max_logit = np.max(logits)
min_logit = np.min(logits)

print(f"Highest logit: {max_logit:.4f}")
print(f"Lowest logit: {min_logit:.4f}")


correct = 0
wrong = 0
for xx, yy in zip(X_test, y_test):
    logits = (np.dot(xx, weights) + intercept) % p
    if yy == (logits > 0):
        correct += 1
    else:
        wrong += 1
print(correct, wrong)
