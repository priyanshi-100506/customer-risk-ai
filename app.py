import gradio as gr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split


# ===============================
# CSV Validation
# ===============================

REQUIRED_COLUMNS = [
    "customer_id",
    "orders",
    "spend",
    "returns",
    "reviews",
    "pay_delay",
    "last_purchase"
]


def validate_csv(df):

    missing = []

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing.append(col)

    if missing:
        return False, f"Missing columns: {', '.join(missing)}"

    return True, "CSV Valid"


# ===============================
# Preprocessing
# ===============================

def preprocess(df):

    df = df.copy()

    df["last_purchase"] = pd.to_datetime(df["last_purchase"])

    df["recency"] = (
        pd.Timestamp.now() - df["last_purchase"]
    ).dt.days

    df.fillna(0, inplace=True)

    return df


# ===============================
# Segmentation
# ===============================

def segment(df):

    features = df[["recency", "orders", "spend"]]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=20
    )

    df["segment"] = model.fit_predict(X)

    labels = {
        0: "VIP",
        1: "Regular",
        2: "Occasional",
        3: "Low Value"
    }

    df["segment_label"] = df["segment"].map(labels)

    return df


# ===============================
# Risk Model
# ===============================

def risk_model(df):

    features = [
        "returns",
        "pay_delay",
        "recency",
        "orders",
        "reviews"
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.12,
        n_estimators=200,
        random_state=42
    )

    model.fit(X_scaled)

    scores = model.decision_function(X_scaled)

    df["risk_score"] = ((1 - scores) * 100).clip(0, 100)

    return df


# ===============================
# Churn Model
# ===============================

def churn_model(df):

    features = [
        "recency",
        "orders",
        "spend",
        "returns",
        "pay_delay",
        "reviews"
    ]

    X = df[features]

    y = (df["recency"] > 90).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    df["churn_probability"] = model.predict_proba(X)[:, 1]

    return df


# ===============================
# Recommendation
# ===============================

def recommend(row):

    if row["risk_score"] > 80:
        return "Urgent Retention"

    if row["churn_probability"] > 0.65:
        return "Re-engagement Offer"

    if row["segment_label"] == "VIP":
        return "Loyalty Rewards"

    if row["recency"] > 120:
        return "Win-back Campaign"

    return "Monitor"


# ===============================
# Charts
# ===============================

def generate_charts(df):

    figs = []

    # Risk Distribution
    fig1, ax1 = plt.subplots()
    ax1.hist(df["risk_score"], bins=20)
    ax1.set_title("Risk Distribution")
    figs.append(fig1)

    # Churn Distribution
    fig2, ax2 = plt.subplots()
    ax2.hist(df["churn_probability"], bins=20)
    ax2.set_title("Churn Distribution")
    figs.append(fig2)

    # Segment Pie
    fig3, ax3 = plt.subplots()
    seg = df["segment_label"].value_counts()
    ax3.pie(seg, labels=seg.index, autopct="%1.1f%%")
    ax3.set_title("Customer Segments")
    figs.append(fig3)

    # Spend vs Risk
    fig4, ax4 = plt.subplots()
    ax4.scatter(df["spend"], df["risk_score"])
    ax4.set_title("Spend vs Risk")
    ax4.set_xlabel("Spend")
    ax4.set_ylabel("Risk")
    figs.append(fig4)

    # Recency Trend
    fig5, ax5 = plt.subplots()
    ax5.hist(df["recency"], bins=20)
    ax5.set_title("Recency Distribution")
    figs.append(fig5)

    return figs


# ===============================
# Compare Customers
# ===============================

def compare_customers(df, id1, id2):

    if id1 not in df["customer_id"].values:
        return "Customer 1 not found"

    if id2 not in df["customer_id"].values:
        return "Customer 2 not found"

    c1 = df[df["customer_id"] == id1].iloc[0]
    c2 = df[df["customer_id"] == id2].iloc[0]

    report = f"""
Customer Comparison

------------------------

Customer 1: {id1}
Segment: {c1['segment_label']}
Risk: {c1['risk_score']:.1f}
Churn: {c1['churn_probability']:.2f}
Spend: {c1['spend']}
Orders: {c1['orders']}

------------------------

Customer 2: {id2}
Segment: {c2['segment_label']}
Risk: {c2['risk_score']:.1f}
Churn: {c2['churn_probability']:.2f}
Spend: {c2['spend']}
Orders: {c2['orders']}

------------------------

Higher Risk: {"Customer 1" if c1['risk_score']>c2['risk_score'] else "Customer 2"}
Higher Value: {"Customer 1" if c1['spend']>c2['spend'] else "Customer 2"}
"""

    return report.strip()


# ===============================
# Main Pipeline
# ===============================

def analyze(file, cust1, cust2):

    if not file:
        return "Upload CSV", "", "", None, None, None, None, None, None, None

    df = pd.read_csv(file.name)

    valid, msg = validate_csv(df)

    if not valid:
        return msg, "", "", None, None, None, None, None, None, None

    df = preprocess(df)
    df = segment(df)
    df = risk_model(df)
    df = churn_model(df)

    df["recommendation"] = df.apply(recommend, axis=1)

    # Save report
    path = "/tmp/customer_report.csv"
    df.to_csv(path, index=False)

    # Summary
    summary = f"""
Total Customers: {len(df)}
High Risk (>70): {(df['risk_score']>70).sum()}
High Churn (>0.6): {(df['churn_probability']>0.6).sum()}
VIP Customers: {(df['segment_label']=="VIP").sum()}
"""

    # Charts
    figs = generate_charts(df)

    # Comparison
    compare = compare_customers(df, cust1, cust2)

    return (
        "Analysis Completed",
        summary,
        compare,
        figs[0],
        figs[1],
        figs[2],
        figs[3],
        figs[4],
        path
    )


# ===============================
# UI
# ===============================

with gr.Blocks() as app:

    gr.Markdown("# Customer Risk Intelligence Dashboard")

    file = gr.File(label="Upload CSV")

    with gr.Row():

        cust1 = gr.Textbox(
            label="Customer ID 1",
            placeholder="e.g. CUST_001"
        )

        cust2 = gr.Textbox(
            label="Customer ID 2",
            placeholder="e.g. CUST_002"
        )

    btn = gr.Button("Run Analysis")

    status = gr.Textbox(label="Status")

    summary = gr.Textbox(label="Business Summary", lines=5)

    comparison = gr.Textbox(label="Customer Comparison", lines=10)

    chart1 = gr.Plot(label="Risk Distribution")
    chart2 = gr.Plot(label="Churn Distribution")
    chart3 = gr.Plot(label="Segments")
    chart4 = gr.Plot(label="Spend vs Risk")
    chart5 = gr.Plot(label="Recency Distribution")

    download = gr.File(label="Download Full Report")

    btn.click(
        analyze,
        inputs=[file, cust1, cust2],
        outputs=[
            status,
            summary,
            comparison,
            chart1,
            chart2,
            chart3,
            chart4,
            chart5,
            download
        ]
    )


app.launch()
