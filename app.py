import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page Setup
st.set_page_config(page_title="E-Commerce Sales Classifier", page_icon="🛍️", layout="wide")
st.title("🛍️ E-Commerce Order Price Classifier")
st.caption("DecodeLabs Industrial Training — Project 2 | Supervised Learning")
st.markdown("---")


# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data_classification.csv")
    cols_to_drop = ["OrderID", "Date", "CustomerID"]
    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Convert continuous target into discrete categories
    target_col = df_clean.columns[-1]
    df_clean[target_col] = pd.qcut(df_clean[target_col], q=3, labels=["Low", "Medium", "High"])
    return df_clean, target_col


df, target_col = load_data()

# 2. Prepare Features & Model
X_raw = df.drop(columns=[target_col])
y = df[target_col]
X_encoded = pd.get_dummies(X_raw, drop_first=True)


@st.cache_resource
def train_model(X_data, y_data):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_data, y_data)
    return model


model = train_model(X_encoded, y)

# 3. Sidebar User Controls
st.sidebar.header("🎛️ Order Input Attributes")
user_inputs = {}

for col in X_raw.columns:
    if X_raw[col].dtype in ['int64', 'float64']:
        min_v = float(X_raw[col].min())
        max_v = float(X_raw[col].max())
        mean_v = float(X_raw[col].mean())
        user_inputs[col] = st.sidebar.slider(f"{col}", min_v, max_v, mean_v)
    else:
        options = list(X_raw[col].unique())
        user_inputs[col] = st.sidebar.selectbox(f"{col}", options)

# Process Input for Model Inference
input_df = pd.DataFrame([user_inputs])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# 4. Display Real-Time Prediction
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Price Category Prediction")
    prediction = model.predict(input_encoded)[0]
    probabilities = model.predict_proba(input_encoded)[0]

    st.metric(label="Predicted Order Price Category", value=f"{prediction} Value")

    # Probability Graph
    prob_df = pd.DataFrame({
        "Category": model.classes_,
        "Confidence (%)": probabilities * 100
    })

    fig_prob = px.bar(
        prob_df, x="Category", y="Confidence (%)",
        title="Class Probability Distribution",
        text_auto=".1f", color="Category"
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with col2:
    st.subheader("📊 Class Distribution Overview")
    class_counts = y.value_counts().reset_index()
    class_counts.columns = ["Category", "Total Orders"]

    fig_pie = px.pie(
        class_counts, names="Category", values="Total Orders",
        title="Dataset Price Bracket Split",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)