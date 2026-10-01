###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Top level functions that can be used to launch a Process."""

from __future__ import annotations

import time
import typing as t

from aiida.common import InvalidOperation
from aiida.common.lang import type_check
from aiida.common.log import AIIDA_LOGGER
from aiida.manage import manager as manager_module
from aiida.orm import ProcessNode

from .processes.builder import ProcessBuilder
from .processes.functions import FunctionProcess
from .processes.process import Process
from .runners import ResultAndPk, Runner
from .utils import instantiate_process, is_process_scoped, prepare_inputs

if t.TYPE_CHECKING:
    from aiida.manage.manager import Manager

__all__ = ('await_processes', 'run', 'run_get_node', 'run_get_pk', 'submit')

TYPE_RUN_PROCESS = Process | type[Process] | ProcessBuilder
# run can also be process function, but it is not clear what type this should be
TYPE_SUBMIT_PROCESS = Process | type[Process] | ProcessBuilder
LOGGER = AIIDA_LOGGER.getChild('engine.launch')


def _resolve_runner(process: TYPE_RUN_PROCESS, manager: Manager | None, *, broker_submit: bool = False) -> Runner:
    """Resolve the runner to use for the given process and manager.

    If the process is an already constructed :class:`~aiida.engine.processes.process.Process` instance, its runner is
    used. In this case, if an explicit ``manager`` is provided that already has a runner and it does not match the
    runner of the process, an :class:`~aiida.common.exceptions.InvalidOperation` is raised instead of silently
    ignoring either one. Otherwise, the runner is obtained from the manager, defaulting to the global manager.

    :param process: the process to be launched
    :param manager: the manager to use; if ``None``, the global manager returned by
        :func:`aiida.manage.manager.get_manager` is used
    :param broker_submit: whether the runner is required for a ``submit``, in which case the runner of the manager is
        returned since it needs the process controller to submit the process to the broker
    :return: the resolved runner
    :raises aiida.common.exceptions.InvalidOperation: if the process is an instance that is bound to a different
        runner than the one of the explicitly provided manager
    """
    if isinstance(process, Process):
        runner = process.runner
        if manager is not None and manager.runner is not None and runner is not manager.runner:
            raise InvalidOperation(
                'the `process` instance is bound to a different runner than the one of the provided `manager`.'
            )
        return runner

    manager = manager or manager_module.get_manager()

    if broker_submit:
        return manager.get_runner()

    # Note: it is safe to create new local runner here without but the explanation can be found in issue #7353
    return manager.create_runner(communicator=None)


def run(
    process: TYPE_RUN_PROCESS,
    inputs: dict[str, t.Any] | None = None,
    *,
    manager: Manager | None = None,
    **kwargs: t.Any,
) -> dict[str, t.Any]:
    """Run the process with the supplied inputs in a local runner that will block until the process is completed.

    :param process: the process class or process function to run
    :param inputs: the inputs to be passed to the process
    :param manager: the manager whose runner is used to run the process; defaults to the global manager returned by
        :func:`aiida.manage.manager.get_manager`
    :return: the outputs of the process
    """
    runner = _resolve_runner(process, manager)

    return runner.run(process, inputs, **kwargs)


def run_get_node(
    process: TYPE_RUN_PROCESS,
    inputs: dict[str, t.Any] | None = None,
    *,
    manager: Manager | None = None,
    **kwargs: t.Any,
) -> tuple[dict[str, t.Any], ProcessNode]:
    """Run the process with the supplied inputs in a local runner that will block until the process is completed.

    :param process: the process class, instance, builder or function to run
    :param inputs: the inputs to be passed to the process
    :param manager: the manager whose runner is used to run the process; defaults to the global manager returned by
        :func:`aiida.manage.manager.get_manager`
    :return: tuple of the outputs of the process and the process node
    """
    runner = _resolve_runner(process, manager)

    return runner.run_get_node(process, inputs, **kwargs)


def run_get_pk(
    process: TYPE_RUN_PROCESS,
    inputs: dict[str, t.Any] | None = None,
    *,
    manager: Manager | None = None,
    **kwargs: t.Any,
) -> ResultAndPk:
    """Run the process with the supplied inputs in a local runner that will block until the process is completed.

    :param process: the process class, instance, builder or function to run
    :param inputs: the inputs to be passed to the process
    :param manager: the manager whose runner is used to run the process; defaults to the global manager returned by
        :func:`aiida.manage.manager.get_manager`
    :return: tuple of the outputs of the process and process node pk
    """
    runner = _resolve_runner(process, manager)

    return runner.run_get_pk(process, inputs, **kwargs)


def submit(
    process: TYPE_SUBMIT_PROCESS,
    inputs: dict[str, t.Any] | None = None,
    *,
    wait: bool = False,
    wait_interval: int = 5,
    manager: Manager | None = None,
    **kwargs: t.Any,
) -> ProcessNode:
    """Submit the process with the supplied inputs to the daemon immediately returning control to the interpreter.

    .. warning: this should not be used within another process. Instead, there one should use the ``submit`` method of
        the wrapping process itself, i.e. use ``self.submit``.

    .. warning: submission of processes requires ``store_provenance=True``.

    :param process: the process class, instance or builder to submit
    :param inputs: the input dictionary to be passed to the process
    :param wait: when set to ``True``, the submission will be blocking and wait for the process to complete at which
        point the function returns the calculation node.
    :param wait_interval: the number of seconds to wait between checking the state of the process when ``wait=True``.
    :param manager: the manager whose runner is used to submit the process. If specified explicitly, the process node
        is also created in the storage of that manager's profile. Defaults to the global manager returned by
        :func:`aiida.manage.manager.get_manager`.
    :param kwargs: inputs to be passed to the process. This is an alternative to the positional ``inputs`` argument.
    :return: the calculation node of the process
    """
    from aiida.common.docs import URL_NO_BROKER

    inputs = prepare_inputs(inputs, **kwargs)

    # Submitting from within another process requires ``self.submit``` unless it is a work function, in which case the
    # current process in the scope should be an instance of ``FunctionProcess``.
    if is_process_scoped() and not isinstance(Process.current(), FunctionProcess):
        raise InvalidOperation('Cannot use top-level `submit` from within another process, use `self.submit` instead')

    # If a dry run is requested, simply forward to `run`, because it is not compatible with `submit`. We choose for this
    # instead of raising, because in this way the user does not have to change the launcher when testing. The same goes
    # for if `remote_folder` is present in the inputs, which means we are importing an already completed calculation.
    # Builder inputs need to be merged so that detection works for ``submit(builder)`` too.
    if isinstance(process, ProcessBuilder):
        merged_inputs = {**inputs, **process._inputs(prune=True)}
    else:
        merged_inputs = inputs
    if (merged_inputs.get('metadata') or {}).get('dry_run', False) or 'remote_folder' in merged_inputs:
        _, node = run_get_node(process, inputs, manager=manager)
        return node

    current_manager = manager or manager_module.get_manager()
    profile = current_manager.get_profile()

    if profile is not None and profile.process_control_backend == 'core.zeromq':
        daemon_client = current_manager.get_daemon_client()
        # Note: ``is_daemon_running`` only checks for a PID file, so a stale PID from a crashed daemon will let this
        # check pass.
        if not daemon_client.is_daemon_running:
            msg = (
                'Cannot submit because the daemon is not running. The ZeroMQ broker is bundled into the daemon for '
                'this profile, so submission requires `verdi daemon start`. To run the process locally without the '
                'daemon instead, '
                'use `aiida.engine.run` (or `run_get_node`).'
            )
            raise InvalidOperation(msg)

    runner = _resolve_runner(process, manager, broker_submit=True)

    if runner.controller is None:
        raise InvalidOperation(
            'Cannot submit because the runner does not have a process controller, probably because the profile does '
            'not define a broker like RabbitMQ. If a RabbitMQ server is available, the profile can be configured to '
            'use it with `verdi profile configure-broker core.rabbitmq`. Otherwise, use '
            ':meth:`aiida.engine.launch.run` instead to run the process in the local Python interpreter instead of '
            'submitting it to the daemon. '
            f'See {URL_NO_BROKER} for more details.'
        )

    assert runner.persister is not None, 'runner does not have a persister'

    # If a manager was explicitly provided, ensure the process node is created in the storage of its profile. When no
    # manager is provided, the backend is left unspecified to preserve the existing lazy-loading behavior.
    backend = manager.get_profile_storage() if manager is not None else None
    process_inited = instantiate_process(runner, process, backend=backend, **inputs)

    if not process_inited.metadata.store_provenance:
        raise InvalidOperation('cannot submit a process with `store_provenance=False`')

    runner.persister.save_checkpoint(process_inited)
    process_inited.close()

    # Do not wait for the future's result, because in the case of a single worker this would cock-block itself
    runner.controller.continue_process(process_inited.pid, nowait=False, no_reply=True)
    node = process_inited.node

    if not wait:
        return node

    while not node.is_terminated:
        LOGGER.report(
            f'Process<{node.pk}> has not yet terminated, current state is `{node.process_state}`. '
            f'Waiting for {wait_interval} seconds.'
        )
        time.sleep(wait_interval)

    return node


def await_processes(nodes: t.Sequence[ProcessNode], wait_interval: int = 1) -> None:
    """Run a loop until all processes are terminated.

    :param nodes: Sequence of nodes that represent the processes to await.
    :param wait_interval: The interval between each iteration of checking the status of all processes.
    """
    type_check(nodes, (list, tuple))

    if any(not isinstance(node, ProcessNode) for node in nodes):
        raise TypeError(f'`nodes` should be a list of `ProcessNode`s but got: {nodes}')

    start_time = time.time()
    terminated = False

    while not terminated:
        running = [not node.is_terminated for node in nodes]
        terminated = not any(running)
        seconds_passed = time.time() - start_time
        LOGGER.report(f'{running.count(False)} out of {len(nodes)} processes terminated. [{round(seconds_passed)} s]')
        time.sleep(wait_interval)


# Allow one to also use run.get_node and run.get_pk as a shortcut, without having to import the functions themselves
run.get_node = run_get_node  # type: ignore[attr-defined]
run.get_pk = run_get_pk  # type: ignore[attr-defined]
