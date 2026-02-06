# mypy: allow-untyped-defs

import smith


class SobolEngine:
    r"""
    The :class:`smith.quasirandom.SobolEngine` is an engine for generating
    (scrambled) Sobol sequences. Sobol sequences are an example of low
    discrepancy quasi-random sequences.

    This implementation of an engine for Sobol sequences is capable of
    sampling sequences up to a maximum dimension of 21201. It uses direction
    numbers from https://web.maths.unsw.edu.au/~fkuo/sobol/ obtained using the
    search criterion D(6) up to the dimension 21201. This is the recommended
    choice by the authors.

    References:
      - Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
        Journal of Complexity, 14(4):466-489, December 1998.

      - I. M. Sobol. The distribution of points in a cube and the accurate
        evaluation of integrals.
        Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.

    Args:
        dimension (Int): The dimensionality of the sequence to be drawn
        scramble (bool, optional): Setting this to ``True`` will produce
                                   scrambled Sobol sequences. Scrambling is
                                   capable of producing better Sobol
                                   sequences. Default: ``False``.
        seed (Int, optional): This is the seed for the scrambling. The seed
                              of the random number generator is set to this,
                              if specified. Otherwise, it uses a random seed.
                              Default: ``None``

    Examples::

        >>> # xdoctest: +SKIP("unseeded random state")
        >>> soboleng = smith.quasirandom.SobolEngine(dimension=5)
        >>> soboleng.draw(3)
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.7500, 0.2500, 0.2500, 0.2500, 0.7500]])
    """

    MAXBIT = 30
    MAXDIM = 21201

    def __init__(self, dimension, scramble=False, seed=None):
        if dimension > self.MAXDIM or dimension < 1:
            raise ValueError(
                "Supported range of dimensionality "
                f"for SobolEngine is [1, {self.MAXDIM}]"
            )

        self.seed = seed
        self.scramble = scramble
        self.dimension = dimension

        cpu = smith.device("cpu")

        self.sobolstate = smith.zeros(
            dimension, self.MAXBIT, device=cpu, dtype=smith.long
        )
        smith._sobol_engine_initialize_state_(self.sobolstate, self.dimension)

        if not self.scramble:
            self.shift = smith.zeros(self.dimension, device=cpu, dtype=smith.long)
        else:
            self._scramble()

        self.quasi = self.shift.clone(memory_format=smith.contiguous_format)
        self._first_point = (self.quasi / 2**self.MAXBIT).reshape(1, -1)
        self.num_generated = 0

    def draw(
        self,
        n: int = 1,
        out: smith.Tensor | None = None,
        dtype: smith.dtype | None = None,
    ) -> smith.Tensor:
        r"""
        Function to draw a sequence of :attr:`n` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(n, dimension)`.

        Args:
            n (Int, optional): The length of sequence of points to draw.
                               Default: 1
            out (Tensor, optional): The output tensor
            dtype (:class:`smith.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        if dtype is None:
            dtype = smith.get_default_dtype()

        if self.num_generated == 0:
            if n == 1:
                result = self._first_point.to(dtype)
            else:
                result, self.quasi = smith._sobol_engine_draw(
                    self.quasi,
                    n - 1,
                    self.sobolstate,
                    self.dimension,
                    self.num_generated,
                    dtype=dtype,
                )
                result = smith.cat((self._first_point.to(dtype), result), dim=-2)
        else:
            result, self.quasi = smith._sobol_engine_draw(
                self.quasi,
                n,
                self.sobolstate,
                self.dimension,
                self.num_generated - 1,
                dtype=dtype,
            )

        self.num_generated += n

        if out is not None:
            out.resize_as_(result).copy_(result)
            return out

        return result

    def draw_base2(
        self,
        m: int,
        out: smith.Tensor | None = None,
        dtype: smith.dtype | None = None,
    ) -> smith.Tensor:
        r"""
        Function to draw a sequence of :attr:`2**m` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(2**m, dimension)`.

        Args:
            m (Int): The (base2) exponent of the number of points to draw.
            out (Tensor, optional): The output tensor
            dtype (:class:`smith.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        n = 2**m
        total_n = self.num_generated + n
        if not (total_n & (total_n - 1) == 0):
            raise ValueError(
                "The balance properties of Sobol' points require "
                f"n to be a power of 2. {self.num_generated} points have been "
                f"previously generated, then: n={self.num_generated}+2**{m}={total_n}. "
                "If you still want to do this, please use "
                "'SobolEngine.draw()' instead."
            )
        return self.draw(n=n, out=out, dtype=dtype)

    def reset(self):
        r"""
        Function to reset the ``SobolEngine`` to base state.
        """
        self.quasi.copy_(self.shift)
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        r"""
        Function to fast-forward the state of the ``SobolEngine`` by
        :attr:`n` steps. This is equivalent to drawing :attr:`n` samples
        without using the samples.

        Args:
            n (Int): The number of steps to fast-forward by.
        """
        if self.num_generated == 0:
            smith._sobol_engine_ff_(
                self.quasi, n - 1, self.sobolstate, self.dimension, self.num_generated
            )
        else:
            smith._sobol_engine_ff_(
                self.quasi, n, self.sobolstate, self.dimension, self.num_generated - 1
            )
        self.num_generated += n
        return self

    def _scramble(self):
        g: smith.Generator | None = None
        if self.seed is not None:
            g = smith.Generator()
            g.manual_seed(self.seed)

        cpu = smith.device("cpu")

        # Generate shift vector
        shift_ints = smith.randint(
            2, (self.dimension, self.MAXBIT), device=cpu, generator=g
        )
        self.shift = smith.mv(
            shift_ints, smith.pow(2, smith.arange(0, self.MAXBIT, device=cpu))
        )

        # Generate lower triangular matrices (stacked across dimensions)
        ltm_dims = (self.dimension, self.MAXBIT, self.MAXBIT)
        ltm = smith.randint(2, ltm_dims, device=cpu, generator=g).tril()

        smith._sobol_engine_scramble_(self.sobolstate, ltm, self.dimension)

    def __repr__(self):
        fmt_string = [f"dimension={self.dimension}"]
        if self.scramble:
            fmt_string += ["scramble=True"]
        if self.seed is not None:
            fmt_string += [f"seed={self.seed}"]
        return self.__class__.__name__ + "(" + ", ".join(fmt_string) + ")"
