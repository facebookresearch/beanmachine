# Logging

Logging in Bean Machine is provided through the [logging](https://docs.python.org/3/howto/logging.html) module in Python. It is recommended that users get familiar with the basics (logger, handler, levels, etc.) of this module before reading further.

## The Bean Machine Logger

Upon importing the beanmachine module, the base logger `"beanmachine"` is initialized as below.  By default, it saves every message at or above the `WARNING` level to a local file and prints every message at or above the `INFO` level to the console. Users could control the information to be logged by replacing the default handlers with customized ones.

```py
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
file_handler = logging.FileHandler("beanmachine.log")
file_handler.setLevel(logging.INFO)

LOGGER = logging.getLogger("beanmachine")
LOGGER.setLevel(logging.INFO)
LOGGER.handlers.clear()
LOGGER.addHandler(console_handler)
LOGGER.addHandler(file_handler)
```

## Pre-defined Levels and Logger Names

To keep sufficient flexibility and ease of use, Bean Machine provides a list of pre-defined logging levels and nested loggers. Users could lower the level of the handler to log more information, or create a filter by the logger's name to whitelist certain information. Experienced users could also create customized filters based on the message itself.

| Level Name \| Logger Name                    | Numric Value | Type of Information to Log                                   |
| -------------------------------------------- | ------------ | ------------------------------------------------------------ |
| ERROR \| beanmachine.error                   | 40           | Exceptions that are caught and handled with alternative solutions. |
| WARNING \|  beanmachine.warning              | 30           | Gradient calculation returns NaN or Inf.                     |
|                                              |              | Falling back to other proposer.                              |
| INFO \| beanmachine.info                     | 20           | User-provided messages.                                      |
| DEBUG_UPDATES \| beanmachine.debug.updates   | 16           | Node name and value                                          |
|                                              |              | Proposed value                                               |
|                                              |              | Proposal log update                                          |
|                                              |              | Children log update                                          |
|                                              |              | Node log update                                              |
|                                              |              | Accept/Reject                                                |
| DEBUG_PROPOSER \| beanmachine.debug.proposer | 14           | Proposer type                                                |
|                                              |              | Proposer properties                                          |
|                                              |              | Step size                                                    |
| DEBUG_GRAPH \| beanmachine.debug.graph       | 12           | The dependency graph of random variables                     |

The level `INFO` and logger name `"beanmachine.info"` are reserved for users who would like to __print__ any information from their model.

## For Developers

Developers are welcome to add additional loggings following the above pattern. e.g. When a new proposer is added, its associated debugging information should use the level `DEBUG_PROPOSER` and the `"beanmachine.debug.proposer"` logger. If a new logging level must be added, please set the level (`LogLevel` class) and the logger name based on the type of information and the expected frequency of use (higher frequency -> higher level), and update the above table.
