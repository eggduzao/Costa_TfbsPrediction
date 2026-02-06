import argparse

import smith


def run_model(level):
    m = smith.nn.Linear(20, 30)
    input = smith.randn(128, 20)
    with smith.backends.mkl.verbose(level):
        m(input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-level", default=0, type=int)
    args = parser.parse_args()
    try:
        run_model(args.verbose_level)
    except Exception as e:
        print(e)
