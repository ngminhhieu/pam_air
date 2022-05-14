import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(20).reshape((10, 2)), range(10)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)
print(X_train)
print(X_test)