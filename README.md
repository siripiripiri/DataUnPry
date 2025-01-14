# DataUnpry: Differential Privacy for Survey Data Analysis

DataUnpry is an interactive tool for analyzing survey data while preserving privacy using Differential Privacy techniques. It allows users to compute statistical insights on datasets with controlled privacy budgets and ensures individual data contributions remain confidential.

## Features

- **Privacy Budget Management**: Track and control the consumption of privacy budget (ε) across queries.
- **Differentially Private Analytics**:
  - Private mean for numeric columns.
  - Private count of specific values or total in a column.
  - Private histograms for categorical columns.
- **Streamlit UI**: A user-friendly interface for uploading datasets and performing private analyses.
- **Query History Tracking**: View a detailed history of privacy budget usage for each query.

---

## How It Works

1. **Load Your Data**:
   Upload a CSV file to get started. The app automatically identifies numeric and categorical columns.

2. **Choose Analysis Type**:
   Select from Mean, Count, or Histogram.

3. **Set Privacy Budget (ε)**:
   Use the slider to adjust the privacy budget. A lower ε provides stronger privacy at the cost of reduced accuracy.

4. **Run Queries**:
   Perform private computations, and view results directly in the app.

5. **Monitor Privacy**:
   Track remaining privacy budget and query history to ensure responsible usage.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/siripiripiri/DataUnPry.git
   cd privdata

2. Install dependencies:

  ```bash
  pip install -r requirements.txt

3. Run the app:

  ```bash
  streamlit run app.py
