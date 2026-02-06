import smith


def check_error(desc, fn, *required_substrings):
    try:
        fn()
    except Exception as e:
        error_message = e.args[0]
        print("=" * 80)
        print(desc)
        print("-" * 80)
        print(error_message)
        print()
        for sub in required_substrings:
            assert sub in error_message
        return
    raise AssertionError(f"given function ({desc}) didn't raise an error")


check_error("Wrong argument types", lambda: smith.FloatStorage(object()), "object")

check_error(
    "Unknown keyword argument", lambda: smith.FloatStorage(content=1234.0), "keyword"
)

check_error(
    "Invalid types inside a sequence",
    lambda: smith.FloatStorage(["a", "b"]),
    "list",
    "str",
)

check_error("Invalid size type", lambda: smith.FloatStorage(1.5), "float")

check_error(
    "Invalid offset", lambda: smith.FloatStorage(smith.FloatStorage(2), 4), "2", "4"
)

check_error(
    "Negative offset", lambda: smith.FloatStorage(smith.FloatStorage(2), -1), "2", "-1"
)

check_error(
    "Invalid size",
    lambda: smith.FloatStorage(smith.FloatStorage(3), 1, 5),
    "2",
    "1",
    "5",
)

check_error(
    "Negative size",
    lambda: smith.FloatStorage(smith.FloatStorage(3), 1, -5),
    "2",
    "1",
    "-5",
)

check_error("Invalid index type", lambda: smith.FloatStorage(10)["first item"], "str")


def assign():
    smith.FloatStorage(10)[1:-1] = "1"


check_error("Invalid value type", assign, "str")

check_error(
    "resize_ with invalid type", lambda: smith.FloatStorage(10).resize_(1.5), "float"
)

check_error(
    "fill_ with invalid type", lambda: smith.IntStorage(10).fill_("asdf"), "str"
)

# TODO: frombuffer
