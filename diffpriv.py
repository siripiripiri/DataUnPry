import pandas as pd
import numpy as np
from diffprivlib import mechanisms
import streamlit as st
from typing import Dict, List, Tuple, Union
import json
from pathlib import Path

class PrivacyBudget:
    def __init__(self, initial_epsilon: float = 1.0):
        self.initial_epsilon = initial_epsilon
        self.remaining_epsilon = initial_epsilon
        self.query_history = []
    
    def consume_budget(self, epsilon: float, query_name: str) -> bool:
        if epsilon <= self.remaining_epsilon:
            self.remaining_epsilon -= epsilon
            self.query_history.append({
                'query': query_name,
                'epsilon_used': epsilon,
                'remaining_budget': self.remaining_epsilon
            })
            return True
        return False
    
    def get_remaining_budget(self) -> float:
        return self.remaining_epsilon
    
    def get_history(self) -> List[Dict]:
        return self.query_history

class PrivateDataAnalyzer:
    def __init__(self, initial_epsilon: float = 1.0):
        self.data = None
        self.privacy_budget = PrivacyBudget(initial_epsilon)
        self.numeric_columns = []
        self.categorical_columns = []
    
    def load_data(self, data: pd.DataFrame) -> None:
        """Load and classify columns in the dataset."""
        self.data = data
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    def _add_laplace_noise(self, value: float, sensitivity: float, epsilon: float) -> float:
        """Add Laplace noise to a numeric value."""
        mech = mechanisms.Laplace(epsilon=epsilon, sensitivity=sensitivity)
        return mech.randomise(value)
    
    def private_mean(self, column: str, epsilon: float = 0.1) -> Union[float, str]:
        """Calculate differentially private mean of a numeric column."""
        if column not in self.numeric_columns:
            return f"Error: {column} is not a numeric column"
        
        if not self.privacy_budget.consume_budget(epsilon, f"mean_{column}"):
            return "Error: Privacy budget exhausted"
        
        true_mean = self.data[column].mean()
        
        # Improved sensitivity calculation for mean
        # Sensitivity for mean = (max - min) / n
        # We divide by n because changing one person's data can only affect the mean by at most (max-min)/n
        n = len(self.data)
        data_range = self.data[column].max() - self.data[column].min()
        sensitivity = data_range / n
        
        # For debugging
        # st.write(f"Debug Info:")
        # st.write(f"True mean: {true_mean}")
        # st.write(f"Data range: {data_range}")
        # st.write(f"Sensitivity: {sensitivity}")
        # st.write(f"Number of records: {n}")
        
        private_mean = self._add_laplace_noise(true_mean, sensitivity, epsilon)
        return private_mean
    
    def private_count(self, column: str, value: str = None, epsilon: float = 0.1) -> Union[int, str]:
        """Calculate differentially private count of values in a column."""
        if not self.privacy_budget.consume_budget(epsilon, f"count_{column}"):
            return "Error: Privacy budget exhausted"
        
        if value is not None:
            true_count = len(self.data[self.data[column] == value])
        else:
            true_count = len(self.data[column].dropna())
        
        sensitivity = 1  # One person can only affect the count by 1
        private_count = self._add_laplace_noise(true_count, sensitivity, epsilon)
        return max(round(private_count), 0)  # Ensure count is non-negative and rounded
    
    def private_histogram(self, column: str, epsilon: float = 0.1) -> Union[Dict, str]:
        """Create a differentially private histogram for a categorical column."""
        if column not in self.categorical_columns:
            return f"Error: {column} is not a categorical column"
        
        if not self.privacy_budget.consume_budget(epsilon, f"histogram_{column}"):
            return "Error: Privacy budget exhausted"
        
        value_counts = self.data[column].value_counts().to_dict()
        sensitivity = 1  # One person can only affect each bin count by 1
        
        private_counts = {}
        for value, count in value_counts.items():
            private_count = self._add_laplace_noise(count, sensitivity, epsilon / len(value_counts))
            private_counts[value] = max(round(private_count), 0)
        
        return private_counts

def create_streamlit_app():
    st.title("Data Unpry: Differential Privacy for Survey Data Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        analyzer = PrivateDataAnalyzer()
        analyzer.load_data(data)
        
        st.write("### Dataset Overview")
        st.write(f"Number of records: {len(data)}")
        st.write(f"Numeric columns: {analyzer.numeric_columns}")
        st.write(f"Categorical columns: {analyzer.categorical_columns}")
        
        # Analysis options
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Mean", "Count", "Histogram"]
        )
        
        epsilon = st.slider("Privacy Budget (ε)", 0.01, 1.0, 0.1, 0.01)
        
        if analysis_type == "Mean":
            column = st.selectbox("Select Column", analyzer.numeric_columns)
            if st.button("Calculate Private Mean"):
                result = analyzer.private_mean(column, epsilon)
                st.write(f"Private Mean of {column}: {result:.2f}")
        
        elif analysis_type == "Count":
            column = st.selectbox("Select Column", data.columns)
            value = st.text_input("Value to count (leave empty for total count)", "")
            if st.button("Calculate Private Count"):
                result = analyzer.private_count(column, value if value else None, epsilon)
                st.write(f"Private Count: {result}")
        
        elif analysis_type == "Histogram":
            column = st.selectbox("Select Column", analyzer.categorical_columns)
            if st.button("Generate Private Histogram"):
                result = analyzer.private_histogram(column, epsilon)
                st.write("Private Histogram:")
                st.bar_chart(pd.Series(result))
        
        # Privacy budget tracking
        st.write("### Privacy Budget Tracking")
        st.write(f"Remaining privacy budget: {analyzer.privacy_budget.get_remaining_budget():.3f}")
        
        # Query history
        st.write("### Query History")
        history = analyzer.privacy_budget.get_history()
        if history:
            history_df = pd.DataFrame(history)
            st.table(history_df)

if __name__ == "__main__":
    create_streamlit_app()