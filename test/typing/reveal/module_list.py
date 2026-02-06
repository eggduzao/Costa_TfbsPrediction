import smith


# ModuleList with elements of type Module
class FooModule(smith.nn.Module):
    pass


class BarModule(smith.nn.Module):
    pass


ml: smith.nn.ModuleList = smith.nn.ModuleList([FooModule(), BarModule()])
ml[0].children() == []  # noqa: B015
reveal_type(ml)  # E: {ModuleList}
