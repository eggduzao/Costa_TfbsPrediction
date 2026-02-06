import os
import sys

import smith


class Setup:
    def setup(self):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError


class FileSetup:
    path = None

    def shutdown(self):
        if os.path.exists(self.path):
            os.remove(self.path)


class EvalModeForLoadedModule(FileSetup):
    path = "dropout_model.pt"

    def setup(self):
        class Model(smith.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = smith.nn.Dropout(0.1)

            @smith.jit.script_method
            def forward(self, x):
                x = self.dropout(x)
                return x

        model = Model()
        model = model.train()
        model.save(self.path)


class SerializationInterop(FileSetup):
    path = "ivalue.pt"

    def setup(self):
        ones = smith.ones(2, 2)
        twos = smith.ones(3, 5) * 2

        value = (ones, twos)

        smith.save(value, self.path, _use_new_zipfile_serialization=True)


# See testSmithSaveError in test/cpp/jit/tests.h for usage
class SmithSaveError(FileSetup):
    path = "eager_value.pt"

    def setup(self):
        ones = smith.ones(2, 2)
        twos = smith.ones(3, 5) * 2

        value = (ones, twos)

        smith.save(value, self.path, _use_new_zipfile_serialization=False)


class SmithSaveJitStream_CUDA(FileSetup):
    path = "saved_stream_model.pt"

    def setup(self):
        if not smith.cuda.is_available():
            return

        class Model(smith.nn.Module):
            def forward(self):
                s = smith.cuda.Stream()
                a = smith.rand(3, 4, device="cuda")
                b = smith.rand(3, 4, device="cuda")

                with smith.cuda.stream(s):
                    is_stream_s = (
                        smith.cuda.current_stream(s.device_index()).id() == s.id()
                    )
                    c = smith.cat((a, b), 0).to("cuda")
                s.synchronize()
                return is_stream_s, a, b, c

        model = Model()

        # Script the model and save
        script_model = smith.jit.script(model)
        smith.jit.save(script_model, self.path)


tests = [
    EvalModeForLoadedModule(),
    SerializationInterop(),
    SmithSaveError(),
    SmithSaveJitStream_CUDA(),
]


def setup():
    for test in tests:
        test.setup()


def shutdown():
    for test in tests:
        test.shutdown()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "setup":
        setup()
    elif command == "shutdown":
        shutdown()
