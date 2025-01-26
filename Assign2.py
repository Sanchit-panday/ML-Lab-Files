import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_data(independent_file, dependent_file):
    """Load and return independent and dependent variables from CSV files."""
    X = pd.read_csv(independent_file, header=None)
    y = pd.read_csv(dependent_file, header=None).values.ravel()
    return X, y

def plot_cost_function(costs, iterations):
    """Plot the cost function vs. iterations."""
    plt.plot(range(iterations), costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function vs. Iterations")
    plt.grid(True)
    plt.show()

def plot_decision_boundary(X, y, theta):
    """Plot the dataset and decision boundary."""
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label="Data Points")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Dataset and Decision Boundary")

    x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_values = -(theta[1] * x_values + theta[0]) / theta[2]
    plt.plot(x_values, y_values, color='red', label="Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_polynomial_decision_boundary(X, y, model, poly):
    """Plot the dataset and polynomial decision boundary."""
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label="Data Points")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Polynomial Features and Decision Boundary")

    x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    Z = model.predict(poly.transform(np.c_[X1.ravel(), X2.ravel()]))
    Z = Z.reshape(X1.shape)
    
    plt.contour(X1, X2, Z, levels=[0.5], colors='red')
    plt.legend()
    plt.grid(True)
    plt.show()

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    """Compute logistic regression cost."""
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1 / m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    """Perform gradient descent for logistic regression."""
    m = len(y)
    costs = []
    for _ in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient
        costs.append(compute_cost(X, y, theta))
    return theta, costs

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using various metrics."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Confusion Matrix:\n", conf_matrix)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

# File paths (replace with actual paths)
independent_file = r'C:\Users\KIIT\Desktop\ML-assignment\assignment2\logisticX.csv'
dependent_file = r'C:\Users\KIIT\Desktop\ML-assignment\assignment2\logisticY.csv'

# Load dataset
X, y = load_data(independent_file, dependent_file)

print("Sanchit Pandey 22052585")
print("X:")
print(X.head())
print("\ny:")
print(y[:5])

# Train logistic regression using sklearn
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Gradient descent implementation
X_bias = np.c_[np.ones((X.shape[0], 1)), X]
theta_init = np.zeros(X_bias.shape[1])
learning_rate = 0.1
iterations = 50
theta, costs = gradient_descent(X_bias, y, theta_init, learning_rate, iterations)

# Plot cost function
plot_cost_function(costs, iterations)

# Plot dataset and decision boundary
plot_decision_boundary(X_bias, y, theta)

# Polynomial feature transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train logistic regression with polynomial features
model.fit(X_poly, y)

# Plot polynomial decision boundary
plot_polynomial_decision_boundary(X, y, model, poly)

# Evaluate the model
y_pred = model.predict(X_poly)
evaluate_model(y, y_pred)
