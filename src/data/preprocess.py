"""
Preprocessing pipeline for CIC-IDS2017 dataset.

Handles data cleaning, label mapping, time-ordered splitting, feature engineering,
and scaling in a leak-safe manner.
"""
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os
import joblib
import pandas as pd
import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import StandardScaler

from data.features import CANONICAL_FEATURES, validate_features

logger = structlog.get_logger(__name__)


class PreprocessingConfig(BaseModel):
    """Configuration for the preprocessing pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    test_days: List[str] = ['Thursday', 'Friday']
    source_file_col: str = 'source_file'
    label_col: str = 'Label'
    category_map: Dict[str, str] = {
        'BENIGN': 'Benign',
        'DoS Hulk': 'DoS',
        'PortScan': 'PortScan',
        'DDoS': 'DDoS',
        'DoS GoldenEye': 'DoS',
        'FTP-Patator': 'BruteForce',
        'SSH-Patator': 'BruteForce',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'Bot': 'Botnet',
        'Web Attack  Brute Force': 'WebAttack',
        'Web Attack  XSS': 'WebAttack',
        'Infiltration': 'Infiltration',
        'Web Attack  Sql Injection': 'WebAttack',
        'Heartbleed': 'Heartbleed'
    }
    output_dir: Path = Path("data/processed")
    dataset_name: str = "cicids2017"


class PreprocessingResult(BaseModel):
    """Result of the preprocessing pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    scaler_path: Path
    stats: Dict[str, Any]
    label_encoder: Optional[Any] = None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw DataFrame by handling inf, NaN, duplicates, and invalid values."""
    logger.info("Starting DataFrame cleaning", initial_rows=len(df), initial_cols=len(df.columns))
    cleaned_df = df.copy()
    
    # Replace inf and -inf with NaN
    cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop columns where >50% of values are NaN
    threshold = len(cleaned_df) * 0.5
    cols_before = set(cleaned_df.columns)
    cleaned_df.dropna(axis=1, thresh=threshold, inplace=True)
    cols_dropped = cols_before - set(cleaned_df.columns)
    logger.info("Dropped columns with >50% NaN", dropped_cols=list(cols_dropped))
    
    # Impute remaining NaN with median
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        medians = cleaned_df[numeric_cols].median()
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(medians)
    
    # Drop duplicates
    rows_before = len(cleaned_df)
    cleaned_df.drop_duplicates(inplace=True)
    logger.info("Dropped duplicates", dropped_rows=rows_before - len(cleaned_df))
    
    # Remove negative values in size/count columns
    count_cols = [c for c in cleaned_df.columns if 'bytes' in c.lower() or 'packets' in c.lower() or 'length' in c.lower()]
    rows_before = len(cleaned_df)
    for col in count_cols:
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            cleaned_df = cleaned_df[cleaned_df[col] >= 0]
    logger.info("Dropped rows with negative size/count", dropped_rows=rows_before - len(cleaned_df))
    
    logger.info("Cleaning complete", final_rows=len(cleaned_df), final_cols=len(cleaned_df.columns))
    return cleaned_df


def map_labels(df: pd.DataFrame, label_col: str, category_map: Dict[str, str]) -> pd.DataFrame:
    """Map raw labels to major categories and add a binary 'is_attack' column."""
    logger.info("Mapping labels", label_col=label_col)
    mapped_df = df.copy()
    
    if label_col in mapped_df.columns:
        mapped_df[label_col] = mapped_df[label_col].astype(str).str.strip()
        mapped_df['mapped_label'] = mapped_df[label_col].map(category_map).fillna('Unknown')
        
        # 0=Benign, 1=Attack
        mapped_df['is_attack'] = (mapped_df['mapped_label'] != 'Benign').astype(int)
        
        dist = mapped_df['is_attack'].value_counts(normalize=True).to_dict()
        logger.info("Label distribution", dist=dist)
    else:
        logger.warning(f"Label column '{label_col}' not found. Cannot map labels.")
        
    return mapped_df


def time_ordered_split(
    df: pd.DataFrame, 
    source_file_col: str = 'source_file', 
    test_days: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a leak-safe time-ordered split based on the source file/day."""
    if test_days is None:
        test_days = ['Thursday', 'Friday']
        
    logger.info("Performing time-ordered split", test_days=test_days, split_col=source_file_col)
    
    if source_file_col not in df.columns:
        logger.warning(f"Split column '{source_file_col}' not found. Using naive split.")
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    else:
        is_test = df[source_file_col].str.contains('|'.join(test_days), case=False, na=False)
        test_df = df[is_test].copy()
        train_df = df[~is_test].copy()
        
    if 'is_attack' in train_df.columns:
        train_dist = train_df['is_attack'].value_counts(normalize=True).to_dict()
        test_dist = test_df['is_attack'].value_counts(normalize=True).to_dict()
        logger.info("Split complete", 
                    train_size=len(train_df), test_size=len(test_df),
                    train_dist=train_dist, test_dist=test_dist)
    else:
        logger.info("Split complete", train_size=len(train_df), test_size=len(test_df))
        
    return train_df, test_df


def engineer_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Engineer custom features, select canonical features, and handle edge cases."""
    logger.info("Engineering features")
    eng_df = df.copy()
    
    eng_df.columns = eng_df.columns.str.strip()
    
    # Add computed features
    if 'Total Fwd Packets' in eng_df.columns and 'Total Backward Packets' in eng_df.columns:
        eng_df['bytes_ratio'] = eng_df['Total Fwd Packets'] / (eng_df['Total Backward Packets'] + 1)
    else:
        eng_df['bytes_ratio'] = 0.0
        
    if 'Flow Bytes/s' in eng_df.columns and 'Flow Duration' in eng_df.columns:
        eng_df['flow_intensity'] = eng_df['Flow Bytes/s'] * eng_df['Flow Duration']
    else:
        eng_df['flow_intensity'] = 0.0
        
    # Select only canonical feature columns (fill missing with 0)
    for col in feature_cols:
        if col not in eng_df.columns:
            eng_df[col] = 0.0
            
    eng_df = eng_df[feature_cols]
    
    # Convert all to numeric, coercing errors to NaN
    eng_df = eng_df.apply(pd.to_numeric, errors='coerce')
    
    # Fill any remaining NaN with 0
    eng_df.fillna(0, inplace=True)
    
    return eng_df


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    """Fit a StandardScaler on training data features."""
    logger.info("Fitting StandardScaler")
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, feature_cols: List[str]) -> pd.DataFrame:
    """Apply a fitted StandardScaler to DataFrame features."""
    logger.info("Applying StandardScaler")
    scaled_df = df.copy()
    scaled_df[feature_cols] = scaler.transform(df[feature_cols])
    return scaled_df


def run_preprocessing_pipeline(raw_df: pd.DataFrame, config: PreprocessingConfig) -> PreprocessingResult:
    """Orchestrate the full preprocessing pipeline."""
    logger.info("Starting preprocessing pipeline")
    
    # 1. Clean
    cleaned_df = clean_dataframe(raw_df)
    
    # 2. Map Labels
    labeled_df = map_labels(cleaned_df, label_col=config.label_col, category_map=config.category_map)
    
    # 3. Time-ordered split
    train_full, test_full = time_ordered_split(labeled_df, config.source_file_col, config.test_days)
    
    # Extract labels
    y_train = train_full['is_attack'] if 'is_attack' in train_full.columns else pd.Series(dtype=int)
    y_test = test_full['is_attack'] if 'is_attack' in test_full.columns else pd.Series(dtype=int)
    
    # 4. Feature Engineering (applied separately after split to avoid leakage)
    feature_list = list(CANONICAL_FEATURES)
    X_train_eng = engineer_features(train_full, feature_list)
    X_test_eng = engineer_features(test_full, feature_list)
    
    # Validate features
    validate_features(X_train_eng)
    validate_features(X_test_eng)
    
    # 5. Scaling
    scaler = fit_scaler(X_train_eng, feature_list)
    X_train_scaled = apply_scaler(X_train_eng, scaler, feature_list)
    X_test_scaled = apply_scaler(X_test_eng, scaler, feature_list)
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Scaler
    scaler_path = config.output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info("Saved scaler", path=str(scaler_path))
    
    # Save processed data as parquet
    train_out = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1) if not y_train.empty else X_train_scaled
    test_out = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1) if not y_test.empty else X_test_scaled
    
    train_out.to_parquet(config.output_dir / "train.parquet")
    test_out.to_parquet(config.output_dir / "test.parquet")
    logger.info("Saved processed data to parquet")
    
    stats = {
        'train_size': len(X_train_scaled),
        'test_size': len(X_test_scaled),
        'features': feature_list
    }
    
    logger.info("Pipeline complete successfully")
    return PreprocessingResult(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_list,
        scaler_path=scaler_path,
        stats=stats
    )


if __name__ == '__main__':
    logger.info("Running standalone preprocessing test")
    dummy_data = {
        'Flow Duration': [100, 200, 300, 400],
        'Total Fwd Packets': [1, 2, 3, 4],
        'Total Backward Packets': [1, 1, 2, 2],
        'Flow Bytes/s': [10.5, 20.1, -5.0, 40.0],
        'source_file': ['Monday', 'Tuesday', 'Thursday', 'Friday'],
        'Label': ['BENIGN', 'DoS Hulk', 'BENIGN', 'Bot']
    }
    
    df = pd.DataFrame(dummy_data)
    config = PreprocessingConfig(output_dir=Path("data/processed_test"))
    
    try:
        result = run_preprocessing_pipeline(df, config)
        print("Preprocessing successful.")
        print(f"Train shape: {result.X_train.shape}, Test shape: {result.X_test.shape}")
    except Exception as e:
        logger.error("Pipeline failed", error=str(e))
