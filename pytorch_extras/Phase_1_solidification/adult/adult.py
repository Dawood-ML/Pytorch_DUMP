"""
Adult Income Classification - UCI Adult Dataset
Focus : Modular architecture, proper data preprocessing, class imbalance handling

Dataset: https://archive.ics.uci.edu/dataset/2/adult
Download: adult.data and adult.test
Place in: ./data/adult/

METACOGNITIVE: This script teaches production-grade code organization.
We separate concerns: data loading, preprocessing, model, training, evaluation.
This is how you write maintainable ML code, not Jupyter notebook spaghetti.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

####
# Data Pipeline
####
class AdultIncomeDataset(Dataset):
    """
    Custom Dataset class is the Right way to handle data in pytorch.
    Why: Enables batching, shuffling, multiprocessing via DataLoader.
    Alternative : Loading Everything into memory as tensors - BAD because: 
    1. Doesn't scale to large datasets
    2. No lazy laoding
    3. Loses Pytorch's optimized data pipeline
    """
    def __init__(self, data_path:str, is_test: bool = False):

        # Column names (UCI Adult dataset has no header)
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        
        # STEP : Load data with proper handling of whitespace and missing values
        # skipinitialspace = True removes leading spaces in values
        # WHY : UCI data has inconsistent spacing that breaks categorical matching
        self.df = pd.read_csv(data_path, names=columns, sep='.\s*',
                              engine='python', na_values='?', skipinitialspace=True)
        
        # Step : Handle missing values by dropping (simple strategy)
        # Alternative : Imputation (mean/mode/model-based)
        # Why not: For this dataset, missing values are <5%, dropping is cleaner
        self.df.dropna(inplace=True)

        # Step : Seperate features and target
        self.y = self.df['income'].values
        self.X = self.df.drop('income', axis=1)
        print("First 5 X's : \n")
        print(f"Shape of X : {self.X.shape}")
        print(f"Length of X : {len(self.X)}")
        print(self.X.head())
        # Step : Identify Categorical and numerical columns
        self.categorical_cols = self.X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nNames of the categorical columns : {self.categorical_cols}")
        print(f"\nNames of the numerical columns : {self.numerical_cols}")

        # Step  : Encode Categorical variables
        # LabelEncoder for Ordinal, OneHot for Nominal
        # WHY : LabelEncoder here: We'll use embeddings in model (more advanced)
        # Alternative: OneHotEncoding - creates sparse high-dim vectors, less efficient
        self.label_encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])
            self.label_encoders[col] = le
        
        # Step : Encode target variable
        # What It means : Convert '>50K' and '<50K' to 1 and 0
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.y)
        