---
id: logging
title: 'Logging'
sidebar_label: 'Logging'
slug: '/logging'
---

Logging in Bean Machine is provided through the [logging](https://docs.python.org/3/howto/logging.html) module in Python. It is recommended that users get familiar with the basics (logger, handler, levels, etc.) of this module before reading further.

## The Bean Machine Logger

Upon importing the beanmachine module, the base logger `"beanmachine"` is initialized with two handlers such that it saves every message at or above the `WARNING` level to a local file and prints every message at or above the `INFO` level to the console. Users could control the information to be logged by replacing the default handlers with customized ones.

## Log Levels and Sub-Loggers

To keep sufficient flexibility and ease of use, Bean Machine provides multiple sub-loggers under the base logger `"beanmachine"`, such as `"beanmachine.inference"`, `"beanmachine.proposer"`, `"beanmachine.world"`, etc. The name of the sub-logger indicates where the message is generated. Following the convention of Python [logging](https://docs.python.org/3/howto/logging.html), users could modify the logging levels of the base or sub-loggers as they need. Experienced users could also create customized filters based on the message itself.

In Bean Machine, we also offer a `LogLevel` class that maps the level name to its numeric value. The mapping is consistent with the `logging` module with additional levels inserted between `INFO` and `DEBUG` to differentiate different priorities in debugging. The following table provides the level description in details.

| Level Name           | Numric Value | Type of Information to Log                                   |
| ---------------------| ------------ | ------------------------------------------------------------ |
| ERROR                | 40           | Exceptions that are caught and handled with alternative solutions. |
| WARNING              | 30           | Gradient calculation returns NaN or Inf.                     |
|                      |              | Falling back to other proposer.                              |
| INFO                 | 20           | User-provided messages.                                      |
| DEBUG_UPDATES        | 16           | Node name and value                                          |
|                      |              | Proposed value                                               |
|                      |              | Proposal log update                                          |
|                      |              | Children log update                                          |
|                      |              | Node log update                                              |
|                      |              | Accept/Reject                                                |
|                      |              | Proposer type                                                |
|                      |              | Proposer properties                                          |
|                      |              | Step size                                                    |
|                      |              | The dependency graph of random variables                     |
|DEBUG                 | 10           | All logging messages                                         |

The level `INFO` is reserved for users who would like to __print__ any information from their model.

## For Developers

Developers are welcome to add additional loggings following the above pattern. e.g. When a new proposer is added, its associated information should use the `"beanmachine.proposer"` logger. If a new logging level must be added, please set the level (`LogLevel` class) based on the type of information and the expected frequency of use (higher frequency -> higher level), and update the above table.
