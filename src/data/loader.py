"""
Data loader module for the IDS project.
Handles loading and downloading of various IDS datasets.
"""

from typing import Protocol, Type, Dict
import pandas as pd
from pathlib import Path
import structlog
from tqdm import tqdm

logger = structlog.get_logger(__name__)

class DatasetSource(Protocol):
    """Protocol for dataset sources."""
    
    def __init__(self, data_dir: Path) -> None:
        ...
        
    def load(self) -> pd.DataFrame:
        """Load and process the dataset, returning a pandas DataFrame."""
        ...

    def download(self) -> None:
        """Download the dataset or provide instructions if manual download is required."""
        ...

class CICIDSSource:
    """Dataset source implementation for CIC-IDS2017."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw' / 'cicids2017'

    def load(self) -> pd.DataFrame:
        """Loads the CIC-IDS2017 dataset."""
        if not self.raw_dir.exists():
            logger.error(f"Directory not found: {self.raw_dir}")
            raise FileNotFoundError(f"Dataset directory not found: {self.raw_dir}")

        csv_files = list(self.raw_dir.glob('*.csv'))
        if not csv_files:
            logger.warning(f"No CSV files found in {self.raw_dir}")
            return pd.DataFrame()

        dataframes = []
        logger.info(f"Loading {len(csv_files)} files for CIC-IDS2017...")
        for file_path in tqdm(csv_files, desc="Loading CIC-IDS2017 files"):
            try:
                df = pd.read_csv(file_path, encoding='utf-8', errors='replace', low_memory=False)
                # Strip leading/trailing whitespaces from column names
                df.columns = df.columns.str.strip()
                df['source_file'] = file_path.name
                dataframes.append(df)
            except Exception as e:
                logger.error("Error reading file", file=file_path.name, error=str(e))

        if not dataframes:
            return pd.DataFrame()

        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info("CIC-IDS2017 loaded successfully", shape=combined_df.shape)
        
        # Log label distribution if the 'Label' column exists
        if 'Label' in combined_df.columns:
            logger.info("Label distribution", distribution=combined_df['Label'].value_counts().to_dict())
            
        return combined_df

    def download(self) -> None:
        """Provides download instructions for CIC-IDS2017."""
        if self.raw_dir.exists() and any(self.raw_dir.glob('*.csv')):
            logger.info("Files for CIC-IDS2017 already exist, skipping download.")
            return

        print("To download the CIC-IDS2017 dataset:")
        print("1. Official Source (Requires form submission): https://www.unb.ca/cic/datasets/ids-2017.html")
        print("2. Kaggle (Alternative): https://www.kaggle.com/datasets/cicdataset/cicids2017")
        print(f"Once downloaded, place the extracted CSV files in: {self.raw_dir.absolute()}")


class UNSWSource:
    """Dataset source implementation for UNSW-NB15."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw' / 'unsw_nb15'

    def load(self) -> pd.DataFrame:
        """Loads the UNSW-NB15 dataset."""
        if not self.raw_dir.exists():
            logger.error(f"Directory not found: {self.raw_dir}")
            raise FileNotFoundError(f"Dataset directory not found: {self.raw_dir}")

        train_file = self.raw_dir / 'UNSW_NB15_training-set.csv'
        test_file = self.raw_dir / 'UNSW_NB15_testing-set.csv'

        dataframes = []
        if train_file.exists() and test_file.exists():
            logger.info("Loading UNSW-NB15 from train/test split files...")
            try:
                train_df = pd.read_csv(train_file, encoding='utf-8', errors='replace', low_memory=False)
                train_df['is_train'] = True
                
                test_df = pd.read_csv(test_file, encoding='utf-8', errors='replace', low_memory=False)
                test_df['is_train'] = False
                
                dataframes.extend([train_df, test_df])
            except Exception as e:
                logger.error("Error reading train/test split files", error=str(e))
        else:
            logger.info("Train/test split files not found. Falling back to numbered CSVs...")
            csv_files = list(self.raw_dir.glob('UNSW-NB15_*.csv'))
            if not csv_files:
                logger.warning(f"No CSV files found in {self.raw_dir}")
                return pd.DataFrame()
                
            for file_path in tqdm(csv_files, desc="Loading UNSW-NB15 numbered files"):
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', errors='replace', low_memory=False)
                    # We can't distinguish train/test for numbered files easily here, set as None
                    df['is_train'] = None 
                    dataframes.append(df)
                except Exception as e:
                    logger.error("Error reading file", file=file_path.name, error=str(e))

        if not dataframes:
            return pd.DataFrame()

        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info("UNSW-NB15 loaded successfully", shape=combined_df.shape)
        
        # Determine the label column name (usually 'label' or 'attack_cat')
        label_col = 'label' if 'label' in combined_df.columns else ('attack_cat' if 'attack_cat' in combined_df.columns else None)
        if label_col:
            logger.info("Label distribution", distribution=combined_df[label_col].value_counts().to_dict())
            
        return combined_df

    def download(self) -> None:
        """Provides download instructions for UNSW-NB15."""
        if self.raw_dir.exists() and any(self.raw_dir.glob('*.csv')):
            logger.info("Files for UNSW-NB15 already exist, skipping download.")
            return

        print("To download the UNSW-NB15 dataset:")
        print("1. Kaggle: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
        print(f"Once downloaded, place the extracted CSV files in: {self.raw_dir.absolute()}")


DATASET_REGISTRY: Dict[str, Type[DatasetSource]] = {
    'cicids2017': CICIDSSource,
    'unsw_nb15': UNSWSource
}

def get_dataset(name: str, data_dir: Path) -> DatasetSource:
    """Factory function to get a dataset source by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset name: {name}. Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    return DATASET_REGISTRY[name](data_dir)

def load_cicids2017(data_dir: Path) -> pd.DataFrame:
    """Helper function for backward compatibility or direct usage."""
    source = CICIDSSource(data_dir)
    return source.load()

def download_cicids2017(data_dir: Path) -> None:
    """Helper function for backward compatibility or direct usage."""
    source = CICIDSSource(data_dir)
    source.download()

def load_unsw_nb15(data_dir: Path) -> pd.DataFrame:
    """Helper function for backward compatibility or direct usage."""
    source = UNSWSource(data_dir)
    return source.load()

def download_unsw_nb15(data_dir: Path) -> None:
    """Helper function for backward compatibility or direct usage."""
    source = UNSWSource(data_dir)
    source.download()

if __name__ == '__main__':
    data_directory = Path('data')
    
    print("--- CIC-IDS2017 ---")
    cic_source = get_dataset('cicids2017', data_directory)
    cic_source.download()
    
    print("\\n--- UNSW-NB15 ---")
    unsw_source = get_dataset('unsw_nb15', data_directory)
    unsw_source.download()
