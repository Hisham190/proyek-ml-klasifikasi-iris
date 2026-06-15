import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "reports" / "evaluation_results.md"

MODEL_FILES = {
    "Decision Tree": MODEL_DIR / "decision_tree_model.pkl",
    "Random Forest": MODEL_DIR / "random_forest_model.pkl",
}

TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def train_default_models():
    MODEL_DIR.mkdir(exist_ok=True)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=1,
        stratify=iris.target,
    )

    decision_tree = DecisionTreeClassifier(random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    joblib.dump(decision_tree, MODEL_FILES["Decision Tree"])
    joblib.dump(random_forest, MODEL_FILES["Random Forest"])


@st.cache_resource
def load_model(model_name):
    if not MODEL_FILES[model_name].exists():
        train_default_models()
    return joblib.load(MODEL_FILES[model_name])


def load_evaluation_table():
    if not REPORT_PATH.exists():
        return None

    text = REPORT_PATH.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.startswith("|")]
    if len(lines) < 3:
        return None

    table_text = "\n".join(lines[:4])
    rows = []
    for line in table_text.splitlines()[2:]:
        parts = [part.strip() for part in line.strip("|").split("|")]
        rows.append(parts)

    return pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])


st.set_page_config(
    page_title="Klasifikasi Iris",
    page_icon="🌸",
    layout="centered",
)

st.title("Klasifikasi Spesies Bunga Iris")
st.write(
    "Aplikasi ini memprediksi spesies bunga Iris berdasarkan ukuran sepal dan petal. "
    "Model yang digunakan adalah Decision Tree dan Random Forest."
)

with st.expander("Informasi fitur"):
    st.write("Sepal length: panjang sepal bunga dalam cm.")
    st.write("Sepal width: lebar sepal bunga dalam cm.")
    st.write("Petal length: panjang petal bunga dalam cm.")
    st.write("Petal width: lebar petal bunga dalam cm.")

st.subheader("Input Data Bunga")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal length", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal width", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

with col2:
    petal_length = st.number_input("Petal length", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal width", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

model_name = st.selectbox("Pilih model", ["Decision Tree", "Random Forest"])

if st.button("Prediksi"):
    model = load_model(model_name)

    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )

    prediction = model.predict(input_data)[0]
    predicted_label = TARGET_NAMES[int(prediction)]

    st.success(f"Hasil Prediksi: {predicted_label}")
    st.write(f"Model yang digunakan: {model_name}")

st.subheader("Hasil Evaluasi Model")
evaluation_df = load_evaluation_table()
if evaluation_df is not None:
    st.dataframe(evaluation_df, use_container_width=True)
else:
    st.info("File evaluasi belum tersedia. Jalankan train_model.py untuk membuat laporan evaluasi.")

st.caption("Dataset: Iris Dataset dari scikit-learn.")
