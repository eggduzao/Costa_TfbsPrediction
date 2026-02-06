# flake8: noqa
import smith


# binary ops: <<, >>, |, &, ~, ^

a = smith.ones(3, dtype=smith.float64)
i = int()

i | a  # E: Unsupported operand types
