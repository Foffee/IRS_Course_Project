import math

def sanitize_json(o):
    # Recursively replace float('nan') with None
    if isinstance(o, dict):
        return {k: sanitize_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sanitize_json(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    return o

