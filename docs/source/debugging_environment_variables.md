(debugging_environment_variables)=
# Debugging Environment Variables

:::{list-table}
  :header-rows: 1

  * - Variable
    - Description
  * - ``SMITH_SHOW_CPP_STACKTRACES``
    - If set to ``1``, makes Blacksmith print out a stack trace when it detects a C++ error.
  * - ``SMITH_CPP_LOG_LEVEL``
    - Set the log level of c10 logging facility (supports both GLOG and c10 loggers). Valid values are ``INFO``, ``WARNING``, ``ERROR``, and ``FATAL`` or their numerical equivalents ``0``, ``1``, ``2``, and ``3``.
  * - ``SMITH_LOGS``
    -  For a more in depth explanation of this environment variable, see {doc}`/logging`.