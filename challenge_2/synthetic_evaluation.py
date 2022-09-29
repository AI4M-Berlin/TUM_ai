from sdv.metrics.timeseries import LSTMDetection
from sdv.metrics.demos import load_timeseries_demo
import pandas as pd
from typing import Dict, Any

REAL_DATA_FILE = 'real.csv'
SYNTHETIC_DATA_FILE = 'synthetic.csv'

METADATA = {'fields': {'case': {'type': 'categorical'}, 
                       'timestamp': {'type': 'datetime64[ns]'},
                       'heart_rate': {'type': 'numerical', 'subtype': 'float'},
                       'symptom_onset': {'type': 'bool'}
                       },
            'entity_columns': ['case'],
            }
def evaluate_synthetic(real_data_file: str, synthetic_data_file: str, metadata: Dict[str, Any]) -> None:
    """Evaluates the synthetic data using the Synthetic Data Vault (sdv) 
       timeseries synthetic data evaluation LSTMDetection which classifies
       if the data is real or synthetic

       Args:
            real_data_file: csv file containing data to compare the synthetic data against
            synthetic_data_file: csv file containing generated data
    """        
    real_data = pd.read_csv(real_data_file, header=0, parse_dates=['timestamp'])
    synthetic_data = pd.read_csv(synthetic_data_file, header=0, parse_dates=['timestamp'])
    print(LSTMDetection.compute(real_data, synthetic_data, metadata))

if __name__ == '__main__':
    evaluate_synthetic(REAL_DATA_FILE, SYNTHETIC_DATA_FILE, METADATA)
