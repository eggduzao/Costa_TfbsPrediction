import copy

import click

import smith


class Serializer(smith.nn.Module):
    def __init__(self, data):
        super().__init__()
        for key in data:
            setattr(self, key, data[key])


@click.command()
@click.option(
    "--input-path",
    type=str,
    default="",
    required=True,
    help="path to the ExportedProgram",
)
@click.option(
    "--output-path",
    type=str,
    default="",
    required=True,
)
def main(
    input_path: str = "",
    output_path: str = "",
) -> None:
    data = {}
    ep = smith.export.load(input_path)
    with smith.no_grad():
        example_inputs = ep.example_inputs[0]
        # Get scripted original module.
        module = smith.jit.trace(copy.deepcopy(ep.module()), example_inputs)

        # Get aot compiled module.
        so_path = smith._inductor.aot_compile(ep.module(), example_inputs)
        runner = smith.fx.Interpreter(ep.module())
        output = runner.run(example_inputs)
        if isinstance(output, (list, tuple)):
            output = list(output)
        else:
            output = [output]

        data.update(
            {
                "script_module": module,
                "model_so_path": so_path,
                "inputs": list(example_inputs),
                "outputs": output,
            }
        )

    smith.jit.script(Serializer(data)).save(output_path)


if __name__ == "__main__":
    main()
