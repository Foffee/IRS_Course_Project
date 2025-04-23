import pandas as pd

from loaded_datasets import items, item_mappings

def expand_recommended_items(items_df):
	item_ids = items_df['parent_asin'].values
	return items[items['parent_asin'].isin(item_ids)]

def extract_item_fields(items: pd.DataFrame, fields: list[str]):
	return items[fields].to_dict(orient='records')

def load_item_contents(recommended_items, fields: list[str], include_contributor=False):
	loaded_items = extract_item_fields(expand_recommended_items(recommended_items), fields)
	
	if include_contributor:
		for item in loaded_items:
			item['main_contributor'] = recommended_items[recommended_items['parent_asin'] == item['parent_asin']]['main_contributor'].unique()[0]

		return loaded_items
	else:
		return loaded_items

def map_asin_to_indices(item_ids: list[str]) -> list[int]:
	return item_mappings[item_mappings['parent_asin'].isin(item_ids)].index.tolist()
