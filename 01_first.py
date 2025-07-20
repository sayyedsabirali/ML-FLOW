# -------------------------------
# 📦 Import Required Libraries
# -------------------------------
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#  Load and Prepare Dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


#  Define Model Parameters

max_depth = 10
n_estimators = 15


# Local tracking (if using local mlflow UI server)
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

#  Define experiment name (optional)
# mlflow.set_experiment('wine_experiment')

#  Remote tracking with DagsHub (for team collaboration)
import dagshub
dagshub.init(repo_owner='sayyedsabirali', repo_name='ML-FLOW', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/sayyedsabirali/ML-FLOW.mlflow')

# Enable automatic logging for sklearn models
mlflow.autolog()



with mlflow.start_run():

    # Train Random Forest Model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Make Predictions & Calculate Accuracy
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log Parameters & Metrics Manually (redundant if autolog is used)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_metric('accuracy', accuracy)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save and log confusion matrix as artifact
    plt.savefig("Confusion-matrix.png")
    mlflow.log_artifact("Confusion-matrix.png")

    # Log current script (optional)
    mlflow.log_artifact(__file__)

    # Add tags to run
    mlflow.set_tags({
        "Author": "Sabir",
        "Project": "Wine Classification"
    })

    # Log trained model ( avoid with DagsHub unless it supports model registry)
    # mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    # Print accuracy
    # print(f"Model Accuracy: {accuracy:.2f}")
