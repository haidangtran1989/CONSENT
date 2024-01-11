from utilities.config import *


def load_types():
    with open(TYPE_FILE_PATH, "r", encoding="utf-8") as type_file:
        types = [line.strip() for line in type_file.readlines()]
    return types
