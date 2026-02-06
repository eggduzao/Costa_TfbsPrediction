# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import smith
from funcsmith import grad, make_fx
from funcsmith.compile import nnc_jit


def f(x):
    return smith.sin(x).sum()


inp = smith.randn(100)
grad_pt = grad(f)
grad_fx = make_fx(grad_pt)(inp)
grad_nnc = nnc_jit(grad_pt)


def bench(name, f, iters=10000, warmup=3):
    for _ in range(warmup):
        f()
    begin = time.time()
    for _ in range(iters):
        f()
    print(f"{name}: ", time.time() - begin)


bench("Blacksmith: ", lambda: grad_pt(inp))
bench("FX: ", lambda: grad_fx(inp))
bench("NNC: ", lambda: grad_nnc(inp))
