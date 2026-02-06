(_smith_nccl_environment_variables)=
# BLACKSMITH ProcessGroupNCCL Environment Variables

For more information on the environment variables, see [ProcessGroupNCCL Environment Variables](https://github.com/blacksmith/blacksmith/blob/main/smith/csrc/distributed/c10d/ProcessGroupNCCL.hpp).

```{list-table}
:header-rows: 1

* - **Variable**
  - **Description**
* - ``SMITH_NCCL_ASYNC_ERROR_HANDLING``
  - Control how we perform Async Error Handling with NCCL when an exception is observed in watchdog. If set to 0, no handling of asynchronous NCCL errors. If set to 1, aborting NCCL communicator and tearing down process upon error. If set to 2, only abort NCCL communicator and if set to 3, tearing down process without aborting NCCL communicator. By default, it is set to 3.
* - ``SMITH_NCCL_HIGH_PRIORITY``
  - Control whether to use high priority stream for the NCCL communicator.
* - ``SMITH_NCCL_BLOCKING_WAIT``
  - Control whether or not wait() is blocking or non-blocking.
* - ``SMITH_NCCL_DUMP_ON_TIMEOUT``
  - Control whether dumping debug info on watchdog timeout or exception is detected. This variable must be set together with SMITH_NCCL_TRACE_BUFFER_SIZE larger than 0.
* - ``SMITH_NCCL_DESYNC_DEBUG``
  - Control whether Desync Debug is enabled. This is helpful in figuring out the culprit rank of collective desync.
* - ``SMITH_NCCL_ENABLE_TIMING``
  - If set to ``1``, enable recording start-events for all ProcessGroupNCCL collectives, and compute accurate collective timing per-collective.
* - ``SMITH_NCCL_ENABLE_MONITORING``
  - If set to ``1``,enable monitoring thread which aborts the process when the ProcessGroupNCCL Watchdog thread gets stuck and no heartbeat is detected after SMITH_NCCL_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling CUDA/NCCL APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged time than necessary tying up cluster resources.
* - ``SMITH_NCCL_HEARTBEAT_TIMEOUT_SEC``
  - Control the watchdog heartbeat timeout period after which the monitoring thread will abort the process.
* - ``SMITH_NCCL_TRACE_BUFFER_SIZE``
  - The maximum number of events we store in the flight recorder's ring buffer. One event could be the start or end of a collective, for example. Set to 0 to disable the tracebuffer and debugging info dump.
* - ``SMITH_NCCL_TRACE_CPP_STACK``
  - Whether to collect cpp stack traces for flight recorder. Default value is False.
* - ``SMITH_NCCL_COORD_CHECK_MILSEC``
  - Control the interval inside the monitoring thread to check the coordinated signal from other ranks, e.g. to dump the debugging information. Default value is 1000 ms.
* - ``SMITH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC``
  - Control how much extra time we will wait for dumping the debugging info before we exit and throws timeout exception.
* - ``SMITH_NCCL_DEBUG_INFO_TEMP_FILE``
  - The file into which the debugging info would be dumped.
* - ``SMITH_NCCL_DEBUG_INFO_PIPE_FILE``
  - The pipe file to trigger debugging dump manually, write anything into the pipe would trigger the dump.
* - ``SMITH_NCCL_NAN_CHECK``
  - Control whether to enable NAN check for the input, Error would be thrown if NAN is detected.
```
