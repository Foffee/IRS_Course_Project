import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
from collections import defaultdict

from als_model import get_als_model
from get_platform import get_platform


# Read items dataframe
def read_items_dataframe():
	file_path = Path(__file__).parent.parent.parent.parent / 'datasets' / 'slimmed' / 'items.csv'
	return pd.read_csv(file_path)


# Read items dataframe
def read_item_map():
	file_path = Path(__file__).parent.parent.parent.parent / 'datasets' / 'mappings' / 'item_map.csv'
	return pd.read_csv(file_path)


# Read text embeddings array
def read_item_text_embeddings():
	file_path = Path(__file__).parent.parent.parent.parent / 'data_structures' / 'item_text_embeddings.npy'
	return np.load(file_path)


# Given a list of numerical ids for items, the item informations are returned
def get_items(item_ids: list[int]):
	print('get_items', 'item_ids', item_ids)
	
	# ✅ This returns a 1D NumPy array, not a DataFrame
	parent_asins = item_map.loc[item_ids, 'parent_asin'].values

	# ✅ Now this will work
	return items[items['parent_asin'].isin(parent_asins)].copy()


# Given the mapped guest vector, N similar items to each item can be found rating-wise (with their scores)
def find_similar_items_by_ratings(mapped_guest_vector: np.ndarray, N=10):
	personalized_items = als_model.similar_items(mapped_guest_vector, N=N+1)

	recommended_items, sim_scores = personalized_items
	similar_items = list(zip(recommended_items, sim_scores)) # do not include the item itself

	return similar_items 


# Given the similar items and their scores, the top N similar items by weighted scoring can be found
def find_most_similar_items_by_ratings(mapped_guest_history, N=10, num_similar_items_per_item=10, decay_rate=0.3):
	# Get a number of similar items for each item in the history
	# Note: In real life, 3-5x the requested amount is retrieved in case filtering removes a lot of items
	similar_items = find_similar_items_by_ratings(mapped_guest_history, N=num_similar_items_per_item*3)
	
	# weighted_scores will be a dict of item id keys & weighted values
	weighted_scores = defaultdict(float)

	# A penalty given to items of different platforms
	MISMATCHED_PLATFORM_PENALTY = 0.1

	for age, (recommended_items, scores) in enumerate(similar_items):
		# Items further into history should contribute less to the recommended items
		decay = np.exp(-decay_rate * age)

		guest_item_platform = get_platform(mapped_guest_history[age])

		for idx, item in enumerate(recommended_items):
			if item in mapped_guest_history:
				continue

			score = scores[idx] * decay
			item_platform = get_platform(item)
			# print('rating', 'item_platform', item, item_platform)

			# If item platforms conflict, then a penalty is applied
			if (item_platform not in [None, 'unknown']) and (guest_item_platform not in [None, 'unknown']) and item_platform != guest_item_platform:
				# print('rating', item, mapped_guest_history[age], item_platform, guest_item_platform)
				score *= MISMATCHED_PLATFORM_PENALTY

			# Add to weighted scores
			if item not in mapped_guest_history:
				weighted_scores[item] += score

	# Sort by score and return the top N
	top_n = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:N]
	return top_n


# Given the mapped guest vector, the most relevant items are returned (via rating)
def item_collaborative_filtering(mapped_guest_history: list[int], N=10, decay_rate=0.3):
	found_similar_items, scores = zip(*find_most_similar_items_by_ratings(mapped_guest_history, N=N, decay_rate=decay_rate))
	found_similar_items = list(found_similar_items)

	item_df = get_items(found_similar_items)
	item_df['score'] = scores

	return item_df, scores


# Given the mapped guest vector, N similar items to each item can be found content-wise (with their scores)
def similar_items_by_content(item_id: int, N=10):
	sim_item_id = np.dot(item_text_embeddings, item_text_embeddings[item_id]) \
		/ (np.linalg.norm(item_text_embeddings, axis=1) * np.linalg.norm(item_text_embeddings[item_id]))

	top_idx = np.argsort(-sim_item_id)[1:N+1]
	return [top_idx, sim_item_id[top_idx]]


# Given the similar items and their scores, the top N similar items by weighted scoring can be found
def find_similar_items_by_content(mapped_guest_vector: np.ndarray, N=10):
	# e^-5 < 0.01 so decay for these items would be very high => items are not taken as significant
	personalized_items = [similar_items_by_content(item_id, N=N) for item_id in mapped_guest_vector[:5]]
	return personalized_items


# Given the similar items and their scores, the top N similar items by weighted scoring can be found
def find_most_similar_items_by_content(mapped_guest_history, N=10, num_similar_items_per_item=10, decay_rate=0.3, diversity_strength=0.5):
	similar_items = find_similar_items_by_content(mapped_guest_history, N=num_similar_items_per_item * 3)
	weighted_scores = defaultdict(float)

	for age, (recommended_items, scores) in enumerate(similar_items):
		decay = np.exp(-decay_rate * age)
		guest_item = mapped_guest_history[age]
		guest_platform = get_platform(int(guest_item))  # 🛠️ Ensure ID is int

		for idx, item in enumerate(recommended_items):
			if item in mapped_guest_history:
				continue

			item_platform = get_platform(int(item))  # 🛠️ Ensure ID is int

			# ✅ LOG the platforms you're comparing
			if item_platform != guest_platform:
				print(f"Skipping: item {item} ({item_platform}) != guest {guest_item} ({guest_platform})")
			
			if (
				item_platform not in ['unknown', None] and
				guest_platform not in ['unknown', None] and
				item_platform != guest_platform
			):
				continue

			score = scores[idx] * decay
			weighted_scores[item] += score

	# Filter and sort top candidates
	candidates = [(item, score) for item, score in weighted_scores.items() if score > 0]
	candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

	# Apply diversity-aware re-ranking
	item_ids = [item for item, _ in candidates]
	reranked = diversity_rerank(item_ids, top_k=N)

	# Return only top-N after reranking
	id_to_score = dict(candidates)
	top_n = [(item, id_to_score[item]) for item in reranked]

	# Debug print
	print("[Final Filtered Content]")
	for item, score in top_n:
		print(f"Item {item} — Platform: {get_platform(int(item))} — Score: {score:.4f}")

	return top_n


# Given the mapped guest vector, the most relevant items are returned (via content)
def content_collaborative_filtering(mapped_guest_history: list[int], N=10, decay_rate=0.3):	
	found_similar_items, scores = zip(*find_most_similar_items_by_content(mapped_guest_history, N=N, decay_rate=decay_rate, diversity_strength=0.9))
	found_similar_items = list(found_similar_items)

	items_df = get_items(found_similar_items)
	items_df['score'] = scores

	return items_df, scores


# Find rating similarity scores of item_id with the given items
def find_rating_similarity_scores(item_id: int, item_indices: list[int]):
	sim_item_id, scores = als_model.similar_items(itemid=item_id, items=item_indices)
	return [sim_item_id[1:], scores[1:]]


# Find content similarity scores of item_id with the given items
def find_content_similarity_scores(item_id: int, item_indices: list[int]):
	sim_item_id = np.dot(item_text_embeddings[item_indices], item_text_embeddings[item_id]) \
		/ (np.linalg.norm(item_text_embeddings[item_indices], axis=1) * np.linalg.norm(item_text_embeddings[item_id]))

	top_idx = np.argsort(-sim_item_id)
	return [item_indices[top_idx], sim_item_id[top_idx]]


# The weight given to items found by ratings changes dynamically as guest history length changes
# Essentially, content based items are given high weights at the beginning but lessen as more items are received
def dynamic_rating_weight(history_len, max_history=5):
	return min(0.8, history_len / max_history)  # max out at 0.8 weight


def diversity_rerank(item_ids: list[int], top_k=10, group_size=3, step_size=4):
	print('\n\n\n', 'diversity_rerank', 'items', get_items(item_ids), '\n\n\n')

	selected = []
	i = 0

	while i < len(item_ids) and len(selected) < top_k:
		group = item_ids[i:i + group_size]
		selected.extend(group)
		i += step_size  # move ahead by step_size (can overlap with previous group)

	return selected[:top_k]


# Get items dataset
items = read_items_dataframe()

# Load the text embeddings for the items
item_text_embeddings = read_item_text_embeddings()

# Load the item mappings
item_map = read_item_map()

# Load the compressed ALS model from its file
als_model = get_als_model()

def preprocess_guest_history(guest_items: list[str]):
	# History would be reverse of the stored guest items
	guest_history = guest_items[::-1]

	# The unique items from the history are collected (in order)
	unique_guest_history = [item for idx, item in enumerate(guest_history) if idx == guest_history.index(item)]

	# Mapping parent_asin of items to numerical ids distrupts the order so a longer approach is used to avoid this
	guest_history_mapped = item_map[item_map['parent_asin'].isin(unique_guest_history)].copy()

	guest_history_mapped['order'] = pd.Categorical(
		guest_history_mapped['parent_asin'],
		categories=unique_guest_history,
		ordered=True
	)

	# The mapped guest history is now ready to be used by item and content collaborative filtering
	guest_history_mapped = guest_history_mapped.sort_values('order').drop(columns='order').index
	return guest_history_mapped

def recommender(guest_items: list[str], N=10, similar_per_item_rating=10, similar_per_item_content=10, rating_decay_rate=0.3, content_decay_rate=0.1, rating_weight=None):
	# Get the mapped guest history ordered from most recent to least
	guest_history_mapped = preprocess_guest_history(guest_items)

	# Get most similar items by rating and content
	similar_items_by_rating, rating_scores = item_collaborative_filtering(guest_history_mapped, N=similar_per_item_rating, decay_rate=rating_decay_rate)
	similar_items_by_content, content_scores = content_collaborative_filtering(guest_history_mapped, N=similar_per_item_content, decay_rate=content_decay_rate)

	# Create two DataFrames to keep track of similarity scores
	df_rating = similar_items_by_rating[['parent_asin', 'title', 'score']].rename(columns={'score': 'score_rating'})
	df_content = similar_items_by_content[['parent_asin', 'title', 'score']].rename(columns={'score': 'score_content'})

	print('df_rating', df_rating)
	print('df_content', df_content)

	df_rating['score_rating'] = rating_scores
	df_content['score_content'] = content_scores

	# Merge recommendations (outer to include all)
	merged_df = pd.merge(df_rating, df_content, on=['parent_asin', 'title'], how='outer')
	merged_df.fillna(0, inplace=True)

	# Drop any items that had 0 for both scores — likely filtered/irrelevant
	merged_df = merged_df[~((merged_df['score_rating'] == 0) & (merged_df['score_content'] == 0))]

	# score_rating or score_content may be 0 if the item did not appear in both recommendations
	# so they must be calculated before the top N recommendations are taken
	# def fill_rating_score(row):
	# 	numerical_parent_asin = item_map[item_map['parent_asin'] == row['parent_asin']].index[0]
	# 	item_platform = get_platform(numerical_parent_asin)

	# 	max_score = 0
	# 	for guest_item in guest_history_mapped:
	# 		guest_platform = get_platform(guest_item)
	# 		score = find_content_similarity_scores(numerical_parent_asin, np.array([guest_item]))[1][0]
	# 		if (item_platform != guest_platform) and (item_platform not in ['unknown', None]) and (guest_platform not in ['unknown', None]):
	# 			score *= 0.0  # or use your MISMATCHED_PLATFORM_PENALTY
	# 		max_score = max(max_score, score)
	# 	return max_score if (row['score_rating'] == 0 or pd.isna(row['score_rating'])) else row['score_rating']

	# def fill_content_score(row):
	# 	numerical_parent_asin = item_map[item_map['parent_asin'] == row['parent_asin']].index[0]
	# 	item_platform = get_platform(numerical_parent_asin)

	# 	max_score = 0
	# 	for guest_item in guest_history_mapped:
	# 		guest_platform = get_platform(guest_item)
	# 		score = find_content_similarity_scores(numerical_parent_asin, np.array([guest_item]))[1][0]
	# 		if (item_platform != guest_platform) and (item_platform not in ['unknown', None]) and (guest_platform not in ['unknown', None]):
	# 			score *= 0.0  # or use your MISMATCHED_PLATFORM_PENALTY
	# 		max_score = max(max_score, score)
	# 	return max_score if (row['score_content'] == 0 or pd.isna(row['score_content'])) else row['score_content']

	# merged_df['score_rating'] = merged_df.apply(fill_rating_score, axis=1)
	# merged_df['score_content'] = merged_df.apply(fill_content_score, axis=1)

	# Normalize the scores in the DataFrame to remove bias
	merged_df['score_rating'] = (merged_df['score_rating'] - merged_df['score_rating'].min()) / (merged_df['score_rating'].max() - merged_df['score_rating'].min() + 1e-6)
	merged_df['score_content'] = (merged_df['score_content'] - merged_df['score_content'].min()) / (merged_df['score_content'].max() - merged_df['score_content'].min() + 1e-6)

	# The final score for each item is calculated via weighted average of the scores and the top N is taken afterwards
	RATING_WEIGHT = rating_weight if rating_weight is not None else dynamic_rating_weight(len(guest_history_mapped))
	CONTENT_WEIGHT = 1 - RATING_WEIGHT

	merged_df['final_score'] = (RATING_WEIGHT * merged_df['score_rating'] + CONTENT_WEIGHT * merged_df['score_content'])
	merged_df = merged_df.sort_values(by='final_score', ascending=False)

	top_n_rating = df_rating.sort_values(by='score_rating', ascending=False).iloc[:N]
	top_n_content = df_content.sort_values(by='score_content', ascending=False).iloc[:N]

	def explain_contributor(row):
		r = RATING_WEIGHT * row['score_rating']
		c = CONTENT_WEIGHT * row['score_content']
		return 'both' if abs(r - c) < 0.05 else ('rating' if r > c else 'content')

	top_n_hybrid = merged_df.iloc[:N]
	top_n_hybrid = top_n_hybrid.assign(
		main_contributor=top_n_hybrid.apply(explain_contributor, axis=1)
	)

	return top_n_rating, top_n_content, top_n_hybrid
