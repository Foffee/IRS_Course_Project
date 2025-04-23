from loaded_datasets import items, item_mappings

def get_platform(item_id: int) -> str:
    # Convert numerical item id to string parent_asin
    parent_asin = item_mappings.loc[item_id]['parent_asin']
    print('test mapping', item_id, parent_asin)

    # Get the item from the items dataset
    item = items[items['parent_asin'] == parent_asin]

    # Get the title of the item
    title = item['title'].iloc[0].lower()

    # Return platform depending on the item
    if 'ps4' in title or 'playstation' or 'ps 4' in title:
        return 'ps4'
    elif 'xbox' in title:
        return 'xbox'
    elif 'nintendo' in title or 'switch' in title:
        return 'nintendo'
    else:
        return 'unknown'
    