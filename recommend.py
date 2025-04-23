from items import load_item_contents

from recommender import recommender

def recommend(guest_data: list[str]):
	# If there is no guest history, then personalization cannot be done
	# and no (personalized) recommendations can be made at this point
	if len(guest_data) == 0:
		return []
	
	# Get personalized recommendations
	top_n_rating, top_n_content, top_n_hybrid = recommender(guest_data)

	# The fields we would like to extract from the item dataframes
	FIELDS = [
		'title', 'features', 'description',
		'details', 'images', 'parent_asin', 
		'categories', 'average_rating', 'rating_number', 
		'main_category', 'store', 'price'
	]

	# Final personalized recommendations are returned
	return {
		'top_n_rating':  load_item_contents(top_n_rating, FIELDS),
		'top_n_content': load_item_contents(top_n_content, FIELDS),
		'top_n_hybrid':  load_item_contents(top_n_hybrid, FIELDS, include_contributor=True),
	}
