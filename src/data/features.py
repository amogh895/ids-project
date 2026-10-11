"""
Feature schema definition and dataset alignment module.

Defines the canonical feature space used across the IDS project to allow models
to operate on both CIC-IDS2017 and UNSW-NB15 datasets uniformly.
"""
from typing import Tuple
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

class FeatureValidationError(Exception):
    """Raised when a DataFrame fails feature validation."""
    pass

CANONICAL_FEATURES: Tuple[str, ...] = (
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'Fwd PSH Flags',
    'SYN Flag Count',
    'RST Flag Count',
    'ACK Flag Count',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'min_seg_size_forward',
    'Active Mean',
    'Idle Mean',
    'bytes_ratio',
    'flow_intensity'
)

# Mapping of UNSW-NB15 columns to their canonical equivalents
UNSW_TO_CANONICAL: dict[str, str] = {
    'dur': 'Flow Duration',
    'spkts': 'Total Fwd Packets',
    'dpkts': 'Total Backward Packets',
    'smean': 'Fwd Packet Length Mean',
    'dmean': 'Bwd Packet Length Mean',
    'sbytes': 'Flow Bytes/s', # Approximation
    'sinpkt': 'Fwd IAT Mean',
    'dinpkt': 'Bwd IAT Mean',
    'synack': 'SYN Flag Count',
    'ackdat': 'ACK Flag Count'
}

def align_to_canonical(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Align a dataset's features to the canonical feature space.
    
    Args:
        df: Raw dataset DataFrame
        dataset_name: Name of the dataset ('cicids2017' or 'unsw_nb15')
        
    Returns:
        DataFrame with exactly the canonical columns in order.
    """
    logger.info("Aligning features to canonical space", dataset_name=dataset_name, initial_shape=df.shape)
    
    df_aligned = df.copy()
    
    if dataset_name.lower() == 'unsw_nb15':
        rename_map = {k: v for k, v in UNSW_TO_CANONICAL.items() if k in df_aligned.columns}
        df_aligned.rename(columns=rename_map, inplace=True)
    elif dataset_name.lower() == 'cicids2017':
        df_aligned.columns = df_aligned.columns.str.strip()
    else:
        logger.warning(f"Unknown dataset name '{dataset_name}'.")
        
    # Ensure all canonical columns exist
    for col in CANONICAL_FEATURES:
        if col not in df_aligned.columns:
            logger.debug(f"Adding missing canonical column '{col}' with zeros.")
            df_aligned[col] = 0.0
            
    # Retain only canonical columns in exact order
    df_aligned = df_aligned[list(CANONICAL_FEATURES)]
    
    logger.info("Alignment complete", final_shape=df_aligned.shape)
    return df_aligned

def validate_features(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame conforms to the canonical feature space.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid
        
    Raises:
        FeatureValidationError: If validation fails
    """
    # Check exact columns in order
    if list(df.columns) != list(CANONICAL_FEATURES):
        missing = set(CANONICAL_FEATURES) - set(df.columns)
        extra = set(df.columns) - set(CANONICAL_FEATURES)
        raise FeatureValidationError(f"Column mismatch. Missing: {missing}. Extra: {extra}.")
        
    # Check for NaN
    if df.isna().any().any():
        raise FeatureValidationError("DataFrame contains NaN values.")
    
    # Check for inf
    if np.isinf(df.select_dtypes(include=[np.number])).any().any():
        raise FeatureValidationError("DataFrame contains infinite values.")
        
    # Check all numeric
    non_numeric = df.select_dtypes(exclude=[np.number, bool]).columns
    if len(non_numeric) > 0:
        raise FeatureValidationError(f"DataFrame contains non-numeric columns: {list(non_numeric)}")
        
    return True
