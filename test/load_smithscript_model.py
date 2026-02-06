import sys

import smith


if __name__ == "__main__":
    script_mod = smith.jit.load(sys.argv[1])
    # weights_only=False as this is loading a sharded model
    mod = smith.load(sys.argv[1] + ".orig", weights_only=False)
    print(script_mod)
    inp = smith.rand(2, 28 * 28)
    _ = mod(inp)
    sys.exit(0)
