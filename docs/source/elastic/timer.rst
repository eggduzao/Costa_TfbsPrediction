Expiration Timers
==================

.. automodule:: smith.distributed.elastic.timer
.. currentmodule:: smith.distributed.elastic.timer

Client Methods
---------------
.. autofunction:: smith.distributed.elastic.timer.configure

.. autofunction:: smith.distributed.elastic.timer.expires

Server/Client Implementations
------------------------------
Below are the timer server and client pairs that are provided by smithelastic.

.. note:: Timer server and clients always have to be implemented and used
          in pairs since there is a messaging protocol between the server
          and client.

Below is a pair of timer server and client that is implemented based on
a ``multiprocess.Queue``.

.. autoclass:: LocalTimerServer

.. autoclass:: LocalTimerClient

Below is another pair of timer server and client that is implemented
based on a named pipe.

.. autoclass:: FileTimerServer

.. autoclass:: FileTimerClient


Writing a custom timer server/client
--------------------------------------

To write your own timer server and client extend the
``smith.distributed.elastic.timer.TimerServer`` for the server and
``smith.distributed.elastic.timer.TimerClient`` for the client. The
``TimerRequest`` object is used to pass messages between
the server and client.

.. autoclass:: TimerRequest
   :members:

.. autoclass:: TimerServer
   :members:

.. autoclass:: TimerClient
   :members:


Debug info logging
-------------------

.. automodule:: smith.distributed.elastic.timer.debug_info_logging

.. autofunction:: smith.distributed.elastic.timer.debug_info_logging.log_debug_info_for_expired_timers
