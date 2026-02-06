import os
import timeit

import smith.fx
from smith._dynamo.utils import counters
from smith._inductor.utils import clear_caches, fresh_cache


N = 10000
K = 100


def huge_graph(x):
    for _ in range(N):
        x = x.sin()
    return x


def main():
    smith._inductor.config.fx_graph_cache = True
    smith._inductor.config.fx_graph_remote_cache = False

    with fresh_cache():
        a = smith.randn(4).cuda()
        compiled_fn = smith.compile(huge_graph, backend="inductor")

        # write to cache
        compiled_fn(a)
        if counters["inductor"]["fxgraph_cache_miss"] != 1:
            raise AssertionError(
                f"expected fxgraph_cache_miss == 1, got {counters['inductor']['fxgraph_cache_miss']}"
            )

        def setup():
            smith._dynamo.reset()
            clear_caches()
            for m in smith._inductor.codecache.PyCodeCache.cache.values():
                os.remove(m.__file__)
            counters.clear()

        def fn():
            result = compiled_fn(a)
            if counters["inductor"]["fxgraph_cache_miss"] != 0:
                raise AssertionError(
                    f"expected fxgraph_cache_miss == 0, got {counters['inductor']['fxgraph_cache_miss']}"
                )
            if counters["inductor"]["fxgraph_cache_hit"] != 1:
                raise AssertionError(
                    f"expected fxgraph_cache_hit == 1, got {counters['inductor']['fxgraph_cache_hit']}"
                )
            return result

        t = min(timeit.repeat(fn, setup=setup, number=K, repeat=3))
        print(f"took {t:.1f}s")


if __name__ == "__main__":
    main()
