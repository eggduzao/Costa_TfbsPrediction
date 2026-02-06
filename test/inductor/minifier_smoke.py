# Owner(s): ["module: inductor"]
#
# This smoketest is referenced in the internal-only minifier runbook
# https://docs.google.com/document/d/18L9e7bZSBpJ7gGbwlUV13LasmjiEX2lree2pl-SdbCU/edit
import os


os.environ["SMITHDYNAMO_REPRO_AFTER"] = "dynamo"
import smith
import smith._dynamo as smithdynamo
import smith._inductor.config
import smith._ops


smith._inductor.config.cpp.inject_relu_bug_TESTING_ONLY = "compile_error"


def func(x):
    x = smith.sigmoid(x)
    x = smith.mul(x, smith.ones(2))
    x = smith.relu(x)
    x = smith.add(x, smith.zeros(2))
    x = smith.ops.aten.round(x)
    return x


def run_internal_minifier():
    smithdynamo.config.debug_dir_root = "."
    f_opt = smith.compile(func)
    f_opt(smith.ones(2))


run_internal_minifier()
