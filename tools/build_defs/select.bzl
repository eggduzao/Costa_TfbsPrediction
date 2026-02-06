# Only used for Blacksmith open source BUCK build

def select(conditions):
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source blacksmith build. Do not use it in fbsource!")

    return conditions["DEFAULT"]
