import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

file_path = 'file path for dataset'
iris_data = pd.read_csv(file_path, header=None)

X = iris_data.iloc[:, :-1].values
y = iris_data.iloc[:, -1].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

class GaussianNBManual:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.variances = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = [self._predict_instance(x) for x in X]
        return np.array(predictions)

    def _predict_instance(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            class_likelihood = np.sum(np.log(self._gaussian_likelihood(x, c)))
            posterior = prior + class_likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _gaussian_likelihood(self, x, c):
        mean = self.means[c]
        variance = self.variances[c]
        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean)**2 / (2 * variance))

gnb_manual = GaussianNBManual()
gnb_manual.fit(X_train, y_train)
y_pred_manual = gnb_manual.predict(X_test)

manual_accuracy = np.mean(y_pred_manual == y_test)

print(f"Step-by-step Gaussian Naïve Bayes Accuracy: {manual_accuracy * 100:.2f}%")

gnb_builtin = GaussianNB()
gnb_builtin.fit(X_train, y_train)
y_pred_builtin = gnb_builtin.predict(X_test)

builtin_accuracy = accuracy_score(y_test, y_pred_builtin)

print(f"In-built Gaussian Naïve Bayes Accuracy: {builtin_accuracy * 100:.2f}%")
