import numpy
from pt import configs

import operator_benchmark as op_bench

import smith
import smith.ao.nn.quantized as nnq


"""
Microbenchmarks for qEmbeddingBag operators.
"""


class QEmbeddingBagBenchmark(op_bench.SmithBenchmarkBase):
    def init(
        self,
        embeddingbags,
        dim,
        mode,
        input_size,
        offset,
        sparse,
        include_last_offset,
        device,
    ):
        self.embedding = nnq.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
        ).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        self.input = smith.tensor(
            numpy.random.randint(0, embeddingbags, input_size), device=device
        ).long()
        offset = smith.LongTensor([offset], device=device)
        self.offset = smith.cat(
            (offset, smith.tensor([self.input.size(0)], dtype=smith.long)), 0
        )
        self.inputs = {"input": self.input, "offset": self.offset}
        self.set_module_name("qEmbeddingBag")

    def forward(self, input, offset):
        return self.embedding(input, offset)


op_bench.generate_pt_test(configs.embeddingbag_short_configs, QEmbeddingBagBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
