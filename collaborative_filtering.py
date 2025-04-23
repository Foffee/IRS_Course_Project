from loaded_datasets import item_mappings, items
from als_model import get_als_model

from items import load_item_contents, map_asin_to_indices
from build_user_vector import build_guest_user_vector

# Load ALS model
als_model = get_als_model()

# Helper function for getting item(s) information after recommendation is made
def get_information_on_items(item_ids, recommendations):
	personalized_items, scores = recommendations
	similar_items = list(zip(personalized_items, scores))

	# similar_items has the integer ids paired with scores
	# We would like those integer ids to be the parent_asin strings instead
	similar_items_mapped = [[load_item_contents(item_mappings.loc[group.tolist()[1:]]['parent_asin'].values.tolist()), group_scoring.tolist()[1:]] for group, group_scoring in similar_items]

	# The similar items (parent_asin and scoring from the original items) are returned
	return {parent_asin:similar_items_mapped[idx] for idx, parent_asin in enumerate(item_ids)}


### Item-based colloborative filtering

def find_base_recommended_items(guest_data: list[str], N=10) -> dict[str, list[str, int]]:
	# Convert parent_asin strings into their respective integer ids
	guest_data_indices = map_asin_to_indices(guest_data)

	# Build the guest user vector
	guest_user_vector = build_guest_user_vector(guest_data_indices, items.shape[0])

	# Use those ids in the ALS model
	# The output will be ids of similar items with their scorings (N+1 since first item is always the item itself)
	personalized_items_scores = als_model.recommend(userid=0, user_items=guest_user_vector, N=N+1, filter_items=guest_data_indices, recalculate_user=True)

	personalized_items, scores = personalized_items_scores
	personalized_items_ids = item_mappings.loc[personalized_items.tolist()[1:]]['parent_asin'].values.tolist()

	loaded_items = load_item_contents(personalized_items_ids)

	recommended_items_mapped = [{item['parent_asin']:item for item in loaded_items}, scores.tolist()[1:]]
	return recommended_items_mapped


def find_similar_items(guest_data: list[str], N=10) -> dict[str, list[list[str], list[float]]]:
	# Convert parent_asin strings into their respective integer ids
	guest_data_indices = map_asin_to_indices(guest_data)

	# Use those ids in the ALS model
	# The output will be ids of similar items with their scorings (N+1 since first item is always the item itself)
	personalized_items_scores = als_model.similar_items(itemid=guest_data_indices, N=N+1)

	return get_information_on_items(guest_data, personalized_items_scores)
