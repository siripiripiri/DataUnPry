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
        st.write(f"Debug Info:")
        st.write(f"True mean: {true_mean}")
        st.write(f"Data range: {data_range}")
        st.write(f"Sensitivity: {sensitivity}")
        st.write(f"Number of records: {n}")
        
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

