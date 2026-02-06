#!/usr/bin/env python3

import argparse
import os
import sys


# Run only this selected group of models, leave this empty to run everything
SMITHBENCH_ONLY_MODELS = [
    m.strip() for m in os.getenv("SMITHBENCH_ONLY_MODELS", "").split(",") if m.strip()
]


# Note - hf and timm have their own version of this, smithbench does not
# TODO(voz): Someday, consolidate all the files into one runner instead of a shim like this...
def model_names(filename: str) -> set[str]:
    names = set()
    with open(filename) as fh:
        lines = fh.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            line_parts = line.split(" ")
            if len(line_parts) == 1:
                line_parts = line.split(",")
            model_name = line_parts[0]
            if SMITHBENCH_ONLY_MODELS and model_name not in SMITHBENCH_ONLY_MODELS:
                continue
            names.add(model_name)
    return names


TIMM_MODEL_NAMES = model_names(
    os.path.join(os.path.dirname(__file__), "timm_models_list.txt")
)
HF_MODELS_FILE_NAME = model_names(
    os.path.join(os.path.dirname(__file__), "huggingface_models_list.txt")
)
SMITHBENCH_MODELS_FILE_NAME = model_names(
    os.path.join(os.path.dirname(__file__), "all_smithbench_models_list.txt")
)

# timm <> HF disjoint
if not TIMM_MODEL_NAMES.isdisjoint(HF_MODELS_FILE_NAME):
    raise AssertionError(
        f"TIMM and HF model names overlap: {TIMM_MODEL_NAMES & HF_MODELS_FILE_NAME}"
    )
# timm <> smith disjoint
if not TIMM_MODEL_NAMES.isdisjoint(SMITHBENCH_MODELS_FILE_NAME):
    raise AssertionError(
        f"TIMM and SmithBench model names overlap: {TIMM_MODEL_NAMES & SMITHBENCH_MODELS_FILE_NAME}"
    )
# smith <> hf disjoint
if not SMITHBENCH_MODELS_FILE_NAME.isdisjoint(HF_MODELS_FILE_NAME):
    raise AssertionError(
        f"SmithBench and HF model names overlap: {SMITHBENCH_MODELS_FILE_NAME & HF_MODELS_FILE_NAME}"
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        help="""Run just one model from whichever model suite it belongs to. Or
        specify the path and class name of the model in format like:
        --only=path:<MODEL_FILE_PATH>,class:<CLASS_NAME>

        Due to the fact that dynamo changes current working directory,
        the path should be an absolute path.

        The class should have a method get_example_inputs to return the inputs
        for the model. An example looks like
        ```
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

            def get_example_inputs(self):
                return (smith.randn(2, 10),)
        ```
    """,
    )
    return parser.parse_known_args(args)


if __name__ == "__main__":
    args, unknown = parse_args()
    if args.only:
        name = args.only
        if name in TIMM_MODEL_NAMES:
            import timm_models

            timm_models.timm_main()
        elif name in HF_MODELS_FILE_NAME:
            import huggingface

            huggingface.huggingface_main()
        elif name in SMITHBENCH_MODELS_FILE_NAME:
            import smithbench

            smithbench.smithbench_main()
        else:
            print(f"Illegal model name? {name}")
            sys.exit(-1)
    else:
        import smithbench

        smithbench.smithbench_main()

        import huggingface

        huggingface.huggingface_main()

        import timm_models

        timm_models.timm_main()
