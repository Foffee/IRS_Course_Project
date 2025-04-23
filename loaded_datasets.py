from pathlib import Path

import pandas as pd

def get_item_mappings() -> pd.Series:
	file_path = Path(__file__).parent.parent.parent.parent / 'datasets' / 'mappings' / 'item_map.csv'
	return pd.read_csv(file_path)

def get_items() -> pd.Series:
	file_path = Path(__file__).parent.parent.parent.parent / 'datasets' / 'slimmed' / 'items.csv'
	return pd.read_csv(file_path)

item_mappings = get_item_mappings()
items = get_items()
