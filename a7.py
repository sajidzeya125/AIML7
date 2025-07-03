import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

# 1. Load and prepare dataset
df = pd.read_csv('breast-cancer.csv')

# Display first few rows
print(df.head())

# Drop non-numeric or irrelevant columns (adjust as needed)
df.dropna(inplace=True)  # remove missing values
X = df.drop(columns=['diagnosis', 'id'])  # change column names if different
y = df['diagnosis'].map({'M': 1, 'B': 0})  # convert labels to binary

# 2. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2 features for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 3. Train SVM with linear and RBF kernel
models = {
    'Linear SVM': SVC(kernel='linear'),
    'RBF SVM': SVC(kernel='rbf')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# 4. Visualization
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.show()

for name, model in models.items():
    plot_decision_boundary(model, X_pca, y, f"{name} Decision Boundary")

# 5. Hyperparameter tuning for RBF SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001]
}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

print("\nBest Parameters (RBF):", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# 6. Evaluate best model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("\nTuned RBF SVM Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
plot_decision_boundary(best_model, X_pca, y, "Tuned RBF SVM Decision Boundary")
