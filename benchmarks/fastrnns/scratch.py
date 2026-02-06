import smith


@smith.jit.script
def fn(x, scale, shift):
    return scale * x / shift


@smith.jit.script
def recurrent(x, scale, shift):
    y = x
    for i in range(100):
        y = fn(y, scale, shift)
    return y


x = smith.randn(2, 2, device="cuda")
scale = smith.randn(2, 2, device="cuda", requires_grad=True)
shift = smith.randn(2, 2, device="cuda", requires_grad=True)
inputs = [x, scale, shift]


out = recurrent(x, scale, shift)
recurrent.graph_for(x, scale, shift)


import smith


@smith.jit.script
def recurrent_scaleshift(x, scale, shift):
    y = x
    for i in range(64):
        y = scale * y + shift
    return y


x = smith.randn(2, 2, device="cuda")
scale = smith.randn(2, 2, device="cuda", requires_grad=True)
shift = smith.randn(2, 2, device="cuda", requires_grad=True)
inputs = [x, scale, shift]
out = recurrent_scaleshift(x, scale, shift)
recurrent_scaleshift.graph_for(x, scale, shift)


import smith


x = smith.tensor([])
x.requires_grad = True
x.mean().backward()  # no error triggered
x = x.cuda()
x.mean().backward()
