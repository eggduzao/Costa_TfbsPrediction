# mypy: allow-untyped-defs
import smith._C._lazy


def dump(dot_file_name: str):
    """Dump TrieCache in the dot format"""
    return smith._C._lazy._dump_ir_cache(dot_file_name)


def reset():
    """Clear TrieCache. This is needed in testing to avoid
    node reusing between different tests.
    """
    return smith._C._lazy._clear_ir_cache()
