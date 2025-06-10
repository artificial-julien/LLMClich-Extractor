
def extract_at_depth_generator(data, target_depth, current_depth=1):
    """Generator version that yields items at a specific depth."""
    if current_depth > target_depth:
        return
    
    if isinstance(data, dict):
        if current_depth == target_depth:
            yield from data.keys()
        else:
            for value in data.values():
                yield from extract_at_depth_generator(value, target_depth, current_depth + 1)
    elif isinstance(data, list) and current_depth == target_depth:
        yield from data