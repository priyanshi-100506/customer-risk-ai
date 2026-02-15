# Requirements

This document lists all **functional, non-functional, and data requirements** for the Customer Risk & Insight AI project.

---

## 1. Functional Requirements

### 1.1 Data Ingestion
- Accept CSV upload of customer data.
- Validate CSV file format and required columns.
- Provide error messages for invalid data.

Required columns:
- `customer_id`
- `orders`
- `spend`
- `returns`
- `reviews`
- `pay_delay`
- `last_purchase`

### 1.2 Analytics & Modeling
- Segment customers using clustering (KMeans).
- Compute risk scores using Isolation Forest.
- Predict churn probability using a classification model.
- Generate AI explanations for customer behavioral insights.
- Recommend business actions based on model outputs.

### 1.3 User Interactions
- Analyze risk for a selected customer or two customers (comparison).
- Provide textual profiles and AI-based summaries.
- Display interactive charts and distribution plots.
- Allow download of the full analyzed report as a CSV.

---

## 2. Non-Functional Requirements

### 2.1 Performance
- Models and analytics should run in under 5 seconds per dataset.
- Charts should render smoothly without timeouts.

### 2.2 Reliability
- Fallback logic if AI APIs fail.
- Graceful error messages in the UI.

### 2.3 Portability
- Runs entirely on Hugging Face Spaces or any Python/Gradio hosting.
- Minimal configuration to deploy.

### 2.4 Security
- API keys stored securely using environment variables.
- No hardcoded credentials.

---

## 3. Technical Stack Requirements

### Backend
- Python 3.8+
- Pandas
- NumPy
- scikit-learn
- matplotlib

### AI/LLM
- OpenAI API (optional, with fallback)
- Environment variable: `OPENAI_API_KEY`

### Frontend
- Gradio

### Visualization
- Matplotlib

---

## 4. External Integrations

| Integration | Purpose | Required |
|-------------|---------|----------|
| OpenAI API | Generate AI explanations | Optional |
| Hugging Face Spaces | Hosting | Yes |

---

## 5. Deployment Requirements

- A `requirements.txt` listing all dependencies.
- Continuous deployment via GitHub if linked with HF Space.
- Clear documentation for judges or collaborators.

