import smith


NUM_REPEATS = 1000
NUM_REPEAT_OF_REPEATS = 1000


class SubTensor(smith.Tensor):
    pass


class WithSmithFunction:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, smith.Tensor):
            self._tensor = data
            return

        self._tensor = smith.tensor(data, requires_grad=requires_grad)

    @classmethod
    def __smith_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        return WithSmithFunction(args[0]._tensor + args[1]._tensor)


class SubWithSmithFunction(smith.Tensor):
    @classmethod
    def __smith_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        return super().__smith_function__(func, types, args, kwargs)
