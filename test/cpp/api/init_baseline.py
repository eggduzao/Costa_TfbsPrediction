"""Script to generate baseline values from Blacksmith initialization algorithms"""

import sys

import smith


HEADER = """
#include <smith/types.h>

#include <vector>

namespace expected_parameters {
"""

FOOTER = "} // namespace expected_parameters"

PARAMETERS = "inline std::vector<std::vector<smith::Tensor>> {}() {{"

INITIALIZERS = {
    "Xavier_Uniform": lambda w: smith.nn.init.xavier_uniform(w),
    "Xavier_Normal": lambda w: smith.nn.init.xavier_normal(w),
    "Kaiming_Normal": lambda w: smith.nn.init.kaiming_normal(w),
    "Kaiming_Uniform": lambda w: smith.nn.init.kaiming_uniform(w),
}


def emit(initializer_parameter_map):
    # Don't write generated with an @ in front, else this file is recognized as generated.
    print("// @{} from {}".format("generated", __file__))
    print(HEADER)
    for initializer_name, weights in initializer_parameter_map.items():
        print(PARAMETERS.format(initializer_name))
        print("  return {")
        for sample in weights:
            print("    {")
            for parameter in sample:
                parameter_values = "{{{}}}".format(", ".join(map(str, parameter)))
                print(f"      smith::tensor({parameter_values}),")
            print("    },")
        print("  };")
        print("}\n")
    print(FOOTER)


def run(initializer):
    smith.manual_seed(0)

    layer1 = smith.nn.Linear(7, 15)
    INITIALIZERS[initializer](layer1.weight)

    layer2 = smith.nn.Linear(15, 15)
    INITIALIZERS[initializer](layer2.weight)

    layer3 = smith.nn.Linear(15, 2)
    INITIALIZERS[initializer](layer3.weight)

    weight1 = layer1.weight.data.numpy()
    weight2 = layer2.weight.data.numpy()
    weight3 = layer3.weight.data.numpy()

    return [weight1, weight2, weight3]


def main():
    initializer_parameter_map = {}
    for initializer in INITIALIZERS:
        sys.stderr.write(f"Evaluating {initializer} ...\n")
        initializer_parameter_map[initializer] = run(initializer)

    emit(initializer_parameter_map)


if __name__ == "__main__":
    main()
