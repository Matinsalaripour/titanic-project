import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class TitanicFeatureEngineer:
   
    def __init__(self):
        # 1. Imputers
        self.age_imputer = SimpleImputer(strategy='median')
        self.embarked_imputer = SimpleImputer(strategy='most_frequent')
        self.fare_imputer = SimpleImputer(strategy='median')  # For test set
        
        # 2. Encoders (OneHot & Ordinal)
        self.title_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        self.embarked_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        
        # Ordinal encoding for Age Groups (Child < Teen < Adult < Middle_Aged < Elder)
        self.age_categories = ['Child', 'Teen', 'Adult', 'Middle_Aged', 'Elder']
        self.age_ordinal_encoder = OrdinalEncoder(categories=[self.age_categories])
        
        # 3. Age binning thresholds
        self.age_bins = [0, 12, 20, 40, 60, np.inf]
        self.age_labels = self.age_categories
        
        # 4. Placeholders for column names after encoding
        self.title_columns = None
        self.embarked_columns = None

    def _extract_title(self, name):
        # Extract Mr, Mrs, Miss, Master, etc., from the Name column.
        match = re.search(r' ([A-Za-z]+)\.', name)
        return match.group(1) if match else 'Unknown'

    def fit(self, df):
        # --- Fit Imputers ---
        self.age_imputer.fit(df[['Age']])
        self.embarked_imputer.fit(df[['Embarked']])      # Correct column name
        self.fare_imputer.fit(df[['Fare']])
        
        # We need to fit the encoders. To do this, we simulate the pipeline on a temporary DataFrame.
        temp = df.copy()
        
        # Apply imputations temporarily (with .ravel() to avoid ValueError)
        temp['Age'] = self.age_imputer.transform(temp[['Age']]).ravel()
        temp['Embarked'] = self.embarked_imputer.transform(temp[['Embarked']]).ravel()
        temp['Fare'] = self.fare_imputer.transform(temp[['Fare']]).ravel()
        
        # 1. Extract Titles
        temp['Title'] = temp['Name'].apply(self._extract_title)
        
        # 2. Bin Ages (to fit the OrdinalEncoder later)
        temp['AgeGroup'] = pd.cut(temp['Age'], bins=self.age_bins, labels=self.age_labels, right=False)
        
        # --- Fit Encoders ---
        self.title_encoder.fit(temp[['Title']])
        self.embarked_encoder.fit(temp[['Embarked']])
        
        # Fit Ordinal Encoder on AgeGroup.
        age_group_clean = temp['AgeGroup'].fillna('Adult').values.reshape(-1, 1)
        self.age_ordinal_encoder.fit(age_group_clean)
        
        # Store generated column names for later
        self.title_columns = self.title_encoder.get_feature_names_out(['Title'])
        self.embarked_columns = self.embarked_encoder.get_feature_names_out(['Embarked'])
        
        return self

    def transform(self, df, is_train=True):
        
        # Create a copy to avoid modifying the original raw data
        df = df.copy()   # <-- Corrected: added parentheses
        
        # --- 1. Apply Simple Imputers (Missing Values) ---
        df['Age'] = self.age_imputer.transform(df[['Age']]).ravel()
        df['Embarked'] = self.embarked_imputer.transform(df[['Embarked']]).ravel()
        df['Fare'] = self.fare_imputer.transform(df[['Fare']]).ravel()
        
        # --- 2. Convert Sex to Numeric (female=0, male=1) ---
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
        
        # --- 3. Extract Titles from Name ---
        df['Title'] = df['Name'].apply(self._extract_title)
        
        # --- 4. Apply np.log to Ages (Log1p handles age=0 safely) ---
        df['AgeLog'] = np.log1p(df['Age'])  # log(age + 1)
        
        # --- 5. Bin Ages into Categories (Child, Teen, Adult, etc.) ---
        df['AgeGroup'] = pd.cut(df['Age'], bins=self.age_bins, labels=self.age_labels, right=False)
        
        # --- 6. Ordinal Encoder for Age Groups ---
        age_group_values = df['AgeGroup'].fillna('Adult').values.reshape(-1, 1)
        df['AgeGroupOrdinal'] = self.age_ordinal_encoder.transform(age_group_values).ravel()
        
        # --- 7. OneHot Encoder for Title and Embarked ---
        title_encoded = self.title_encoder.transform(df[['Title']])
        embarked_encoded = self.embarked_encoder.transform(df[['Embarked']])
        
        # Convert to DataFrames with proper column names
        title_df = pd.DataFrame(title_encoded, columns=self.title_columns, index=df.index)
        embarked_df = pd.DataFrame(embarked_encoded, columns=self.embarked_columns, index=df.index)
        
        # --- 8. Create Family Size Feature & IsAlone ---
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # --- 9. Drop Unnecessary Columns ---
        cols_to_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Age']
        
        if is_train:
            cols_to_drop.append('PassengerId')
        
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')
        
        # --- 10. Concatenate the encoded columns and drop temporary string columns ---
        df = pd.concat([df, title_df, embarked_df], axis=1)
        df.drop(columns=['Title', 'AgeGroup', 'Embarked'], inplace=True, errors='ignore')
        
        # --- 11. Drop redundant numeric columns (based on correlation analysis) ---
        # AgeLog has 0.89 correlation with AgeGroupOrdinal -> keep AgeGroupOrdinal, drop AgeLog
        # FamilySize has almost zero correlation with Survived -> keep IsAlone, drop FamilySize
        extra_drop = ['AgeLog', 'FamilySize']
        df.drop(columns=[col for col in extra_drop if col in df.columns], inplace=True, errors='ignore')
        
        return df