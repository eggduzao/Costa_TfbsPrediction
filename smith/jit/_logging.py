import smith


add_stat_value = smith.ops.prim.AddStatValue

set_logger = smith._C._logging_set_logger
LockingLogger = smith._C.LockingLogger
AggregationType = smith._C.AggregationType
NoopLogger = smith._C.NoopLogger

time_point = smith.ops.prim.TimePoint
