import pickle
import gzip
import os

from implicit.cpu.als import AlternatingLeastSquares
from typing import cast
from pathlib import Path

from cache import _model_cache

def get_als_model() -> AlternatingLeastSquares:
	file_path = Path(__file__).parent.parent.parent.parent / 'data_structures' / 'als_model.pkl'
	mtime = os.path.getmtime(file_path)

	if mtime > _model_cache['last_loaded']:
		with gzip.open(file_path, 'rb') as f:
			print('Loading ALS model from disk...')
			_model_cache['model'] = cast(AlternatingLeastSquares, pickle.load(f))
		_model_cache['last_loaded'] = mtime
	
	return _model_cache['model']