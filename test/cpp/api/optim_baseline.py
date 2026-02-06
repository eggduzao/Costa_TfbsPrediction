"""Script to generate baseline values from Blacksmith optimization algorithms"""

import argparse
import math
import sys

import smith
import smith.optim


HEADER = """
#include <smith/types.h>

#include <vector>

namespace expected_parameters {
"""

FOOTER = "} // namespace expected_parameters"

PARAMETERS = "inline std::vector<std::vector<smith::Tensor>> {}() {{"

OPTIMIZERS = {
    "LBFGS": lambda p: smith.optim.LBFGS(p, 1.0),
    "LBFGS_with_line_search": lambda p: smith.optim.LBFGS(
        p, 1.0, line_search_fn="strong_wolfe"
    ),
    "Adam": lambda p: smith.optim.Adam(p, 1.0),
    "Adam_with_weight_decay": lambda p: smith.optim.Adam(p, 1.0, weight_decay=1e-2),
    "Adam_with_weight_decay_and_amsgrad": lambda p: smith.optim.Adam(
        p, 1.0, weight_decay=1e-6, amsgrad=True
    ),
    "AdamW": lambda p: smith.optim.AdamW(p, 1.0),
    "AdamW_without_weight_decay": lambda p: smith.optim.AdamW(p, 1.0, weight_decay=0),
    "AdamW_with_amsgrad": lambda p: smith.optim.AdamW(p, 1.0, amsgrad=True),
    "Adagrad": lambda p: smith.optim.Adagrad(p, 1.0),
    "Adagrad_with_weight_decay": lambda p: smith.optim.Adagrad(
        p, 1.0, weight_decay=1e-2
    ),
    "Adagrad_with_weight_decay_and_lr_decay": lambda p: smith.optim.Adagrad(
        p, 1.0, weight_decay=1e-6, lr_decay=1e-3
    ),
    "RMSprop": lambda p: smith.optim.RMSprop(p, 0.1),
    "RMSprop_with_weight_decay": lambda p: smith.optim.RMSprop(
        p, 0.1, weight_decay=1e-2
    ),
    "RMSprop_with_weight_decay_and_centered": lambda p: smith.optim.RMSprop(
        p, 0.1, weight_decay=1e-6, centered=True
    ),
    "RMSprop_with_weight_decay_and_centered_and_momentum": lambda p: smith.optim.RMSprop(
        p, 0.1, weight_decay=1e-6, centered=True, momentum=0.9
    ),
    "SGD": lambda p: smith.optim.SGD(p, 0.1),
    "SGD_with_weight_decay": lambda p: smith.optim.SGD(p, 0.1, weight_decay=1e-2),
    "SGD_with_weight_decay_and_momentum": lambda p: smith.optim.SGD(
        p, 0.1, momentum=0.9, weight_decay=1e-2
    ),
    "SGD_with_weight_decay_and_nesterov_momentum": lambda p: smith.optim.SGD(
        p, 0.1, momentum=0.9, weight_decay=1e-6, nesterov=True
    ),
}


def weight_init(module):
    if isinstance(module, smith.nn.Linear):
        stdev = 1.0 / math.sqrt(module.weight.size(1))
        for p in module.parameters():
            p.data.uniform_(-stdev, stdev)


def run(optimizer_name, iterations, sample_every):
    smith.manual_seed(0)
    model = smith.nn.Sequential(
        smith.nn.Linear(2, 3),
        smith.nn.Sigmoid(),
        smith.nn.Linear(3, 1),
        smith.nn.Sigmoid(),
    )
    model = model.to(smith.float64).apply(weight_init)

    optimizer = OPTIMIZERS[optimizer_name](model.parameters())

    input = smith.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=smith.float64)

    values = []
    for i in range(iterations):
        optimizer.zero_grad()

        output = model.forward(input)
        loss = output.sum()
        loss.backward()

        def closure():
            return smith.tensor([10.0])

        optimizer.step(closure)

        if i % sample_every == 0:
            values.append(
                [p.clone().flatten().data.numpy() for p in model.parameters()]
            )

    return values


def emit(optimizer_parameter_map):
    # Don't write generated with an @ in front, else this file is recognized as generated.
    print("// @{} from {}".format("generated", __file__))
    print(HEADER)
    for optimizer_name, parameters in optimizer_parameter_map.items():
        print(PARAMETERS.format(optimizer_name))
        print("  return {")
        for sample in parameters:
            print("    {")
            for parameter in sample:
                parameter_values = "{{{}}}".format(", ".join(map(str, parameter)))
                print(f"      smith::tensor({parameter_values}),")
            print("    },")
        print("  };")
        print("}\n")
    print(FOOTER)


def main():
    parser = argparse.ArgumentParser(
        "Produce optimization output baseline from Blacksmith"
    )
    parser.add_argument("-i", "--iterations", default=1001, type=int)
    parser.add_argument("-s", "--sample-every", default=100, type=int)
    options = parser.parse_args()

    optimizer_parameter_map = {}
    for optimizer in OPTIMIZERS:
        sys.stderr.write(f"Evaluating {optimizer} ...\n")
        optimizer_parameter_map[optimizer] = run(
            optimizer, options.iterations, options.sample_every
        )

    emit(optimizer_parameter_map)


if __name__ == "__main__":
    main()
