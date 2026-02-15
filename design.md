# System Design & Architecture

This document outlines the **architecture, component design, data flow, and key decisions** for the Customer Risk & Insight AI system.

---

## 1. Overview

The system ingests customer transaction data and generates:

- Customer insights (segmentation & value)
- Risk scores (anomaly detection)
- Churn probabilities
- AI-powered explanations
- Visual analytics & downloadable reports

It is designed for retail and marketplace use.

---

## 2. High-Level Architecture

```
 ┌──────────────────┐
 │   User (Browser) │
 └─────────┬────────┘
           │
       Gradio UI
           │
 ┌─────────▼─────────┐
 │  Application API │
 │ (app.py Backend) │
 └───▲───────────▲───┘
     │           │
     │           └─── Machine Learning Models
     │                • Isolation Forest (Risk)
     │                • RandomForest (Churn)
     │                • KMeans (Segmentation)
     │
     └── AI Component (OpenAI Optional)
          • LLM Explanations
          • Fallback Rule Logic
```

---

## 3. Components

### 3.1 Gradio Frontend
- Takes file uploads and inputs.
- Displays:
  - Customer profile
  - Risk/churn summaries
  - Graphs
  - Comparison panel
  - Download button

### 3.2 Backend Logic
- Data validation
- Preprocessing
- Modeling
- Report generation
- Chart creation

### 3.3 Models

| Model | Purpose | Implementation |
|-------|---------|----------------|
| KMeans | Customer segmentation | scikit-learn |
| IsolationForest | Risk score | scikit-learn |
| RandomForest | Churn prediction | scikit-learn |
| GPT | AI explanation (optional) | OpenAI API |

---

## 4. Data Flow

1. User uploads CSV.
2. System validates columns.
3. Preprocessor transforms recency and features.
4. Analytics models compute segments, risk, churn.
5. Explain logic runs (AI or fallback).
6. UI displays results and charts.
7. Full report saved and downloadable.

---

## 5. Chart Designs

| Chart | Purpose |
|-------|---------|
| Risk Distribution | Visualize overall risk spread |
| Churn Distribution | Probability histogram |
| Segment Pie | Customer breakdown |
| Spend vs Risk | Correlation view |
| Recency | Inactivity trends |

---

## 6. Comparison Logic

Accepts two customer IDs.

Outputs:
- Segment label comparison
- Risk differences
- Churn differences
- Spend/order comparison

---

## 7. Scalability Considerations

- Load testing for large CSVs
- Async model execution (future)
- Caching for repeated requests

---

## 8. Deployment

- Hosted on Hugging Face Spaces
- Dependencies via `requirements.txt`
- Environment variables for secrets

---

## 9. Future Enhancements

- Add PDF reporting export
- Support time-series trend charts
- Add role-based access control
- Plug-in marketplace data sources

