import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_PATH = 'customer_data.csv'
OUTPUT_DIR = Path('processed')
OUTPUT_DIR.mkdir(exist_ok=True)