import os


def get_resource_path(rel):
    """
    :meta private:
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
