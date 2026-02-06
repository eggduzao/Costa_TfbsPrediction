# Only used for Blacksmith open source BUCK build

def expect(condition, message = None):
    if not condition:
        fail(message)
