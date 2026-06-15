from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df.to_csv(DATA_DIR / "iris_dataset.csv", index=False)

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1,
        stratify=y,
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    rows = []
    report_text = "# Evaluation Results\n\n"
    report_text += "Hasil evaluasi model menggunakan data uji sebesar 20 persen dari total dataset.\n\n"

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1-score": f1_score(y_test, y_pred, average="weighted"),
            }
        )

        file_name = model_name.lower().replace(" ", "_") + "_model.pkl"
        joblib.dump(model, MODEL_DIR / file_name)

    result_df = pd.DataFrame(rows)

    report_text += result_df.to_markdown(index=False, floatfmt=".6f")
    report_text += "\n\n"

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        report_text += f"## Classification Report {model_name}\n\n"
        report_text += "```text\n"
        report_text += classification_report(y_test, y_pred, target_names=iris.target_names)
        report_text += "\n```\n\n"

    (REPORT_DIR / "evaluation_results.md").write_text(report_text, encoding="utf-8")

    print("Training selesai.")
    print(result_df.to_string(index=False))
    print("\nModel tersimpan di folder models.")
    print("Hasil evaluasi tersimpan di reports/evaluation_results.md.")


if __name__ == "__main__":
    main()
