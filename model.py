from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the machine learning model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


if __name__ == "__main__":
    model, X_test, y_test = train_model()
    accuracy = evaluate_model(model, X_test, y_test)

    print("Model: Logistic Regression")
    print("Dataset: Iris")
    print(f"Model Accuracy: {accuracy:.2%}")

    # Save accuracy for the GitHub Pages deployment
    with open("accuracy.txt", "w") as file:
        file.write(f"{accuracy:.2%}")
