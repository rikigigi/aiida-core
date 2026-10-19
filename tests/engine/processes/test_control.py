"""Tests for the :mod:`aiida.engine.processes.control` module."""

from unittest import mock

import pytest
from plumpy.process_comms import RemoteProcessThreadController

from aiida.common.exceptions import InvalidOperation
from aiida.engine import Process, ProcessState, launch
from aiida.engine.launch import run, submit
from aiida.engine.processes import control
from aiida.orm import Int
from tests.utils.processes import WaitProcess


@pytest.mark.usefixtures('aiida_profile_clean', 'started_daemon_client')
@pytest.mark.parametrize('action', (control.pause_processes, control.play_processes, control.kill_processes))
def test_processes_all_exclusivity(submit_and_await, action):
    """Test that control methods raise if both ``processes`` is specified and ``all_entries=True``."""
    node = submit_and_await(WaitProcess, ProcessState.WAITING)
    assert not node.paused

    with pytest.raises(ValueError, match='cannot specify processes when `all_entries = True`.'):
        action([node], all_entries=True)


@pytest.mark.usefixtures('aiida_profile_clean', 'stopped_daemon_client')
@pytest.mark.parametrize('action', (control.pause_processes, control.play_processes, control.kill_processes))
def test_daemon_not_running(action, caplog):
    """Test that control methods warns if the daemon is not running."""
    action(all_entries=True)
    assert 'The daemon is not running' in caplog.records[0].message


@pytest.mark.usefixtures('aiida_profile_clean', 'started_daemon_client')
def test_pause_processes(submit_and_await):
    """Test :func:`aiida.engine.processes.control.pause_processes`."""
    node = submit_and_await(WaitProcess, ProcessState.WAITING)
    assert not node.paused

    control.pause_processes([node], timeout=float('inf'))
    assert node.paused
    assert node.process_status == 'Paused through `aiida.engine.processes.control.pause_processes`'


@pytest.mark.usefixtures('aiida_profile_clean', 'started_daemon_client')
def test_pause_processes_all_entries(submit_and_await):
    """Test :func:`aiida.engine.processes.control.pause_processes` with ``all_entries=True``."""
    node = submit_and_await(WaitProcess, ProcessState.WAITING)
    assert not node.paused

    control.pause_processes(all_entries=True, timeout=float('inf'))
    assert node.paused


@pytest.mark.usefixtures('aiida_profile_clean', 'started_daemon_client')
def test_play_processes(submit_and_await):
    """Test :func:`aiida.engine.processes.control.play_processes`."""
    node = submit_and_await(WaitProcess, ProcessState.WAITING)
    assert not node.paused

    control.pause_processes([node], timeout=float('inf'))
    assert node.paused

    control.play_processes([node], timeout=float('inf'))
    assert not node.paused


@pytest.mark.usefixtures('aiida_profile_clean', 'started_daemon_client')
def test_play_processes_all_entries(submit_and_await):
    """Test :func:`aiida.engine.processes.control.play_processes` with ``all_entries=True``."""
    node = submit_and_await(WaitProcess, ProcessState.WAITING)
    assert not node.paused

    control.pause_processes([node], timeout=float('inf'))
    assert node.paused

    control.play_processes(all_entries=True, timeout=float('inf'))
    assert not node.paused


@pytest.mark.usefixtures('aiida_profile_clean', 'started_daemon_client')
def test_kill_processes(submit_and_await):
    """Test :func:`aiida.engine.processes.control.kill_processes`."""
    node = submit_and_await(WaitProcess, ProcessState.WAITING)

    control.kill_processes([node], timeout=float('inf'))
    assert node.is_terminated
    assert node.is_killed
    assert node.process_status == 'Killed through `aiida.engine.processes.control.kill_processes`'


@pytest.mark.usefixtures('aiida_profile_clean', 'started_daemon_client')
def test_kill_processes_all_entries(submit_and_await):
    """Test :func:`aiida.engine.processes.control.kill_processes` with ``all_entries=True``."""
    node = submit_and_await(WaitProcess, ProcessState.WAITING)

    control.kill_processes(all_entries=True, timeout=float('inf'))
    assert node.is_terminated
    assert node.is_killed


@pytest.mark.usefixtures('aiida_profile_clean', 'started_daemon_client')
def test_revive(monkeypatch, aiida_code_installed, submit_and_await):
    """Test :func:`aiida.engine.processes.control.revive_processes`."""
    code = aiida_code_installed(default_calc_job_plugin='core.arithmetic.add', filepath_executable='/bin/bash')
    builder = code.get_builder()
    builder.x = Int(1)
    builder.y = Int(1)
    builder.metadata.options.resources = {'num_machines': 1}

    # Temporarily patch the ``RemoteProcessThreadController.continue_process`` method to do nothing and just return a
    # completed future. This ensures that the submission creates a process node in the database but no task is sent to
    # RabbitMQ and so the daemon will not start running it.
    with monkeypatch.context() as context:
        context.setattr(RemoteProcessThreadController, 'continue_process', lambda *args, **kwargs: None)
        node = submit(builder)

    # The node should now be created in the database but "stuck"
    assert node.process_state == ProcessState.CREATED

    # Time to revive it by recreating the task and send it to RabbitMQ
    control.revive_processes([node])

    # Wait for the process to be picked up by the daemon and completed. If there is a problem with the code, this call
    # should timeout and raise an exception
    submit_and_await(node)
    assert node.is_finished_ok


def make_mock_process(pk=1):
    """Return a mock that satisfies the interface of a ``ProcessNode`` used by the control functions."""
    process = mock.MagicMock(name=f'process<{pk}>')
    process.pk = pk
    process.is_terminated = False
    return process


class TestProcessControlManager:
    """Test that the control functions use the explicitly passed ``manager`` instead of the global manager."""

    @pytest.fixture
    def mock_manager(self, monkeypatch):
        """Return a mock manager and a spy on the module-level ``get_manager`` of the control module."""
        manager = mock.MagicMock(name='manager')
        manager.get_daemon_client.return_value.is_daemon_running = True

        get_manager_spy = mock.MagicMock(name='get_manager')
        monkeypatch.setattr(control, 'get_manager', get_manager_spy)

        return manager, get_manager_spy

    @pytest.mark.parametrize('action', (control.pause_processes, control.play_processes, control.kill_processes))
    def test_explicit_manager_used(self, mock_manager, action):
        """The action should go through the controller of the explicit manager and not consult the global manager."""
        manager, get_manager_spy = mock_manager
        process = make_mock_process()

        action([process], timeout=None, manager=manager)

        assert get_manager_spy.call_count == 0
        controller = manager.get_process_controller.return_value
        method = getattr(controller, f'{action.__name__.replace("_processes", "")}_process')
        method.assert_called_once()

    def test_explicit_manager_used_revive(self, mock_manager):
        """``revive_processes`` should use the explicit manager and not consult the global manager."""
        manager, get_manager_spy = mock_manager
        process = make_mock_process()

        control.revive_processes([process], manager=manager)

        assert get_manager_spy.call_count == 0
        manager.get_process_controller.return_value.continue_process.assert_called_once_with(
            process.pk, nowait=True, no_reply=False
        )

    def test_explicit_manager_daemon_client(self, mock_manager):
        """The daemon status check should use the daemon client of the explicit manager."""
        manager, _ = mock_manager
        manager.get_daemon_client.return_value.is_daemon_running = False
        process = make_mock_process()

        control.play_processes([process], timeout=None, manager=manager)

        manager.get_daemon_client.assert_called_with()

    def test_revive_raises_daemon_not_running(self, mock_manager):
        """``revive_processes`` should raise if the daemon of the explicit manager is not running."""
        manager, _ = mock_manager
        manager.get_daemon_client.return_value.is_daemon_running = False

        with pytest.raises(control.DaemonException, match='The daemon is not running'):
            control.revive_processes([make_mock_process()], manager=manager)

    @pytest.mark.parametrize('action', (control.pause_processes, control.play_processes, control.kill_processes))
    def test_all_entries_uses_manager_storage(self, mock_manager, monkeypatch, action):
        """With ``all_entries=True``, the query for active processes should use the storage of the explicit manager."""
        manager, get_manager_spy = mock_manager
        backend = manager.get_profile_storage.return_value

        builder = mock.MagicMock(name='builder')
        builder.all.return_value = []
        construct = mock.MagicMock(return_value=builder)
        monkeypatch.setattr(control, 'QueryBuilder', construct)

        action(all_entries=True, manager=manager)

        assert get_manager_spy.call_count == 0
        manager.get_profile_storage.assert_called_with()
        assert construct.call_args.kwargs['backend'] is backend


class TestLaunchManager:
    """Test that the ``launch`` functions correctly handle an explicitly passed ``manager``."""

    def test_mismatched_runner_raises(self):
        """Passing a process instance together with a manager that has a different runner should raise."""
        runner = mock.MagicMock(name='runner')
        process = mock.MagicMock(spec=Process, name='process-instance')
        process.runner = runner

        manager = mock.MagicMock(name='manager')
        manager.runner = mock.MagicMock(name='other-runner')

        with pytest.raises(InvalidOperation, match='bound to a different runner'):
            run(process, manager=manager)

        # When the runner of the process matches the runner of the manager, no error is raised and the runner of the
        # process is used.
        manager = mock.MagicMock(name='manager')
        manager.runner = runner

        assert launch._resolve_runner(process, manager) is runner

    def test_runner_obtained_from_manager(self):
        """For process classes, the runner should be obtained from the explicitly passed manager."""
        process_cls = mock.MagicMock(name='process_class')
        manager = mock.MagicMock(name='manager')
        runner = manager.create_runner.return_value

        run(process_cls, manager=manager)

        # Upstream changed `run` to create a new local runner without a communicator, see issue #7353.
        manager.create_runner.assert_called_once_with(communicator=None)
        manager.get_runner.assert_not_called()
        runner.run.assert_called_once_with(process_cls, None)

    def test_default_manager_fallback(self):
        """Without an explicit manager, the global manager should be used."""
        process_cls = mock.MagicMock(name='process_class')
        get_manager_mock = mock.MagicMock(name='get_manager')

        with mock.patch.object(launch.manager_module, 'get_manager', get_manager_mock):
            run(process_cls)

        get_manager_mock.assert_called_once_with()
        # Upstream changed `run` to create a new local runner without a communicator, see issue #7353.
        get_manager_mock.return_value.create_runner.assert_called_once_with(communicator=None)
        get_manager_mock.return_value.get_runner.assert_not_called()
