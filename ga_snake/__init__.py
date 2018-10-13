def _flatten_list(lst):
    if lst is None:
        return []
    return sum(lst, [])


def id_generator():
    uid = 0
    while True:
        yield uid
        uid += 1