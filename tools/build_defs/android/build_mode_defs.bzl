def is_production_build():
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source blacksmith build. Do not use it in fbsource!")
    return False
