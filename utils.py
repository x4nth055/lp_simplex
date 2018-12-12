
def get_all_occ(iterable, value):
    """
    Returns a all index occurences of `value` in the `iterable`
    """
    return [ i for i, k in enumerate(iterable) if k == value ]


def sorter1(k):
    if 'e' in k or 's' in k:
        return 100000 + int(k[1])
    elif 'a' in k or 'A' in k:
        return 200000 + int(k[1])
    else:
        return int(k[1])

def sorter2(k):
    return sorter1(k[0])