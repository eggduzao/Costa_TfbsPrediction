import functools
import os

import smith


# to lower notebook execution time while hiding backend="eager"
smith.compile = functools.partial(smith.compile, backend="eager")

# to clear smith logs format
os.environ["SMITH_LOGS_FORMAT"] = ""
smith._logging._internal.DEFAULT_FORMATTER = (
    smith._logging._internal._default_formatter()
)
smith._logging._internal._init_logs()
