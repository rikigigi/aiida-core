import asyncio
import os
import stat
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiida.transports.plugins.async_backend import _OpenSSH, get_openssh_version
from aiida.transports.plugins.ssh_async import AsyncSshTransport


class TestAuthenticationScript:
    """Tests for authentication script (2FA) functionality."""

    @pytest.fixture
    def make_script(self, tmp_path):
        """Factory to create a script with given exit code."""

        def _make(exit_code=0):
            script = tmp_path / f'script_{exit_code}.sh'
            script.write_text(f'#!/bin/bash\necho "STDOUT"\necho "STDERR" >&2\nexit {exit_code}\n')
            os.chmod(script, 0o755)
            return script

        return _make

    def _make_transport(self, script_path=None, use_old_param=False):
        """Helper to create transport with mocked backend."""
        kwargs = {'machine': 'localhost', 'backend': 'asyncssh'}
        if script_path:
            key = 'script_before' if use_old_param else 'authentication_script'
            kwargs[key] = str(script_path)

        transport = AsyncSshTransport(**kwargs)
        transport.async_backend = MagicMock()
        transport.async_backend.open = AsyncMock()
        transport.async_backend.logger = MagicMock()
        return transport

    @pytest.mark.asyncio
    async def test_script_success(self, make_script):
        """Script returning 0 succeeds and logs info."""
        transport = self._make_transport(make_script(exit_code=0))

        await transport.open_async()

        transport.async_backend.logger.info.assert_called()
        assert 'executed successfully' in transport.async_backend.logger.info.call_args[0][0]

    @pytest.mark.asyncio
    async def test_script_failure_raises_and_logs(self, make_script):
        """Script returning non-zero raises OSError and logs stdout/stderr."""
        transport = self._make_transport(make_script(exit_code=1))

        with pytest.raises(OSError, match='failed with exit code 1'):
            await transport.open_async()

        error_msg = transport.async_backend.logger.error.call_args[0][0]
        assert 'stdout: STDOUT' in error_msg
        assert 'stderr: STDERR' in error_msg

    @pytest.mark.asyncio
    async def test_backward_compatibility_script_before(self, make_script):
        """Old 'script_before' parameter still works."""
        script = make_script(exit_code=0)
        transport = self._make_transport(script, use_old_param=True)
        assert transport.auth_script == str(script)

    @pytest.mark.asyncio
    async def test_no_script_configured(self):
        """Transport works without authentication script."""
        transport = self._make_transport()
        await transport.open_async()

        assert transport.auth_script == 'None'


class TestSemaphoreBehavior:
    """Tests for AsyncSshTransport semaphore/concurrency control."""

    @pytest.mark.asyncio
    async def test_semaphore_released_after_errors(self, tmp_path_factory):
        """Verify semaphore is properly released even when operations fail.

        This ensures that failed I/O operations don't cause semaphore leaks,
        which would eventually deadlock the transport.
        """
        local_dir = tmp_path_factory.mktemp('local')

        # Create a file without read permissions to trigger upload errors
        unreadable_file = local_dir / 'unreadable'
        unreadable_file.write_text('test file without read permissions\n')
        os.chmod(unreadable_file, stat.S_IWGRP | stat.S_IWUSR)  # chmod 220

        transport_params = {
            'machine': 'localhost',
            'backend': 'asyncssh',
            'max_io_allowed': 1,
        }
        async_transport = AsyncSshTransport(**transport_params)

        async with async_transport as transport:
            # Each operation should fail but release the semaphore
            with pytest.raises(OSError, match='Error while downloading file'):
                await transport.getfile_async('non_existing', local_dir)

            with pytest.raises(OSError, match='Error while uploading file'):
                await transport.puttree_async(local_dir, 'target_dir')

            # This would deadlock if semaphore wasn't released after previous errors
            with pytest.raises(OSError, match='Error while downloading file'):
                await transport.getfile_async('non_existing', local_dir)

        assert async_transport._semaphore._value == 1, 'Semaphore should be fully released'

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_operations(self):
        """Verify exec_command_wait_async respects max_io_allowed.

        This is particularly critical, because exec_command_wait_async opens subchannels on the SSH
        connection, and too many simultaneous subchannels can overwhelm the server.
        """
        max_allowed = 2
        transport = AsyncSshTransport(
            machine='localhost',
            backend='asyncssh',
            max_io_allowed=max_allowed,
        )

        concurrent_count = 0
        max_concurrent = 0

        async def mock_run(command, stdin=None, timeout=None):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate I/O latency
            concurrent_count -= 1
            return (0, 'output', '')

        mock_backend = MagicMock()
        mock_backend.run = mock_run
        transport.async_backend = mock_backend

        # Launch more tasks than the semaphore allows
        tasks = [transport.exec_command_wait_async(f'cmd{i}') for i in range(5)]
        await asyncio.gather(*tasks)

        assert (
            max_concurrent <= max_allowed
        ), f'Semaphore should limit to {max_allowed} concurrent ops, got {max_concurrent}'
        assert max_concurrent == max_allowed, f'Expected {max_allowed} concurrent ops to verify semaphore is being used'


def test_get_openssh_version():
    """Test that get_openssh_version() parses version correctly."""
    mock_result = MagicMock()
    mock_result.stdout = ''

    mock_result.stderr = 'OpenSSH_9.0p1, OpenSSL 3.0.2'
    with patch('aiida.transports.plugins.async_backend.subprocess.run', return_value=mock_result):
        assert get_openssh_version() == 9

    mock_result.stderr = 'OpenSSH_8.9p1, OpenSSL 3.0.2'
    with patch('aiida.transports.plugins.async_backend.subprocess.run', return_value=mock_result):
        assert get_openssh_version() == 8

    mock_result.stderr = ''
    with patch('aiida.transports.plugins.async_backend.subprocess.run', return_value=mock_result):
        assert get_openssh_version() is None


class _TestOpenSSH(_OpenSSH):
    """Minimal OpenSSH subclass for testing escape methods."""

    def __init__(self):
        self.machine = 'localhost'
        self.bash_command = 'bash -c '


class TestSshCommandGenerator:
    """Tests for ssh_command_generator escaping."""

    def test_escapes_shell_chars_and_paths(self):
        """Test $, `, " escaped and paths safely quoted."""

        openssh = _TestOpenSSH()

        """Test command structure and that $, `, " are escaped for the outer double-quote wrapper."""
        result = openssh.ssh_command_generator('echo $HOME `cmd` "test"')
        assert result[:2] == ['ssh', 'localhost']
        assert result[2].startswith('bash -c "') and result[2].endswith('"')
        assert r'\$HOME' in result[2]
        assert r'\`cmd\`' in result[2]
        assert r'\"test\"' in result[2]

        """Test $!, $?, $$ are escaped so they're interpreted by inner bash -c shell."""
        result = openssh.ssh_command_generator('script.sh & echo $!; echo $?; echo $$')
        assert r'\$!' in result[2]
        assert r'\$?' in result[2]
        assert r'\$\$' in result[2]

        """Test paths with special chars are safely quoted via escape_for_bash."""
        result = openssh.ssh_command_generator('cp {} {}', paths=['/path;rm -rf /', '/dst'])
        # Semicolon should be safely quoted, not interpreted as command separator
        assert "'/path;rm -rf /'" in result[2]
        assert "'/dst'" in result[2]


def test_escape_for_glob_preserves_wildcards_escapes_dangerous_chars():
    """Test wildcards (*, ?, []) preserved while dangerous chars escaped."""
    backend = _TestOpenSSH()
    # Wildcards preserved, dangerous chars escaped
    assert backend._escape_for_glob('/home/$USER/my files/[0-9]*.log') == '/home/\\$USER/my\\ files/[0-9]*.log'
    # Command injection attempt escaped
    assert backend._escape_for_glob('/path;rm -rf /*') == '/path\\;rm\\ -rf\\ /*'


def test_escape_for_rcp():
    """Test RCP mode escapes shell metacharacters."""
    backend = _TestOpenSSH()
    assert backend._escape_for_rcp('/path/with spaces/$VAR;cmd') == '/path/with\\ spaces/\\$VAR\\;cmd'

    def test_escape_for_scp_with_O_flag(self, openssh_backend):
        """Test that -O flag forces RCP escaping regardless of OpenSSH version."""
        path = '/path/with (parentheses)'
        
        # Set scp_command to include -O flag
        openssh_backend.scp_command = ['scp', '-O']
        
        # Even with OpenSSH 9+, -O flag should force escaping
        with patch('aiida.transports.plugins.async_backend.is_openssh_9_or_higher', return_value=True):
            assert openssh_backend._escape_for_scp(path) == r'/path/with\ \(parentheses\)'  # RCP: escaped

        # With OpenSSH < 9, -O flag should also force escaping
        with patch('aiida.transports.plugins.async_backend.is_openssh_9_or_higher', return_value=False):
            assert openssh_backend._escape_for_scp(path) == r'/path/with\ \(parentheses\)'  # RCP: escaped


def test_escape_for_scp_version_aware():
    """Test _escape_for_scp behavior differs by OpenSSH version."""
    backend = _TestOpenSSH()
    path = '/path/with spaces'

    # SFTP mode (OpenSSH 9+): no escaping needed
    backend.is_openssh_9_or_higher = True
    assert backend._escape_for_scp(path) == path

    # RCP mode (OpenSSH < 9): escaping required
    backend.is_openssh_9_or_higher = False
    assert backend._escape_for_scp(path) == '/path/with\\ spaces'


def test_scp_with_special_chars(tmp_path):
    """Test scp in both SFTP and RCP modes with special characters."""
    backend = _TestOpenSSH()

    remote_dir = tmp_path / 'remote'
    local_dir = tmp_path / 'local'
    remote_dir.mkdir()
    local_dir.mkdir()

    special_files = ['file with spaces.txt', "file'quote.txt", 'file$dollar.txt', 'aiida.pdos_atm#2(Al)_wfc#2(p).txt']
    for f in special_files:
        (remote_dir / f).write_text(f'content of {f}')

    for filename in special_files:
        source = remote_dir / filename

        # SFTP mode (default on OpenSSH 9+)
        dest_sftp = local_dir / f'sftp_{filename}'
        result = subprocess.run(['scp', f'localhost:{source}', str(dest_sftp)], capture_output=True, check=False)
        assert result.returncode == 0 and dest_sftp.read_text() == f'content of {filename}'

        # RCP mode (forced with -O)
        dest_rcp = local_dir / f'rcp_{filename}'
        escaped = backend._escape_for_rcp(str(source))
        result = subprocess.run(['scp', '-O', f'localhost:{escaped}', str(dest_rcp)], capture_output=True, check=False)
        assert result.returncode == 0 and dest_rcp.read_text() == f'content of {filename}'

class TestScpCommandConfiguration:
    """Tests for custom SCP command configuration."""

    def test_default_scp_command(self):
        """Test that default SCP command is ['scp']."""
        transport = AsyncSshTransport(machine='localhost', backend='openssh')
        assert transport.scp_command == ['scp']
        assert transport.async_backend.scp_command == ['scp']

    def test_custom_scp_command(self):
        """Test that custom SCP command is properly set."""
        custom_command = ['scp', '-O']
        transport = AsyncSshTransport(machine='localhost', backend='openssh', scp_command=custom_command)
        assert transport.scp_command == custom_command
        assert transport.async_backend.scp_command == custom_command

    def test_custom_scp_command_with_multiple_options(self):
        """Test that custom SCP command with multiple options works."""
        custom_command = ['scp', '-O', '-v', '-P', '2222']
        transport = AsyncSshTransport(machine='localhost', backend='openssh', scp_command=custom_command)
        assert transport.scp_command == custom_command
        assert transport.async_backend.scp_command == custom_command

    def test_custom_scp_command_with_full_path(self):
        """Test that custom SCP command with full path works."""
        custom_command = ['/usr/bin/scp', '-O']
        transport = AsyncSshTransport(machine='localhost', backend='openssh', scp_command=custom_command)
        assert transport.scp_command == custom_command
        assert transport.async_backend.scp_command == custom_command

    def test_string_scp_command_cli_compatibility(self):
        """Test that string SCP commands are converted to lists for CLI compatibility."""
        # Test simple string
        transport1 = AsyncSshTransport(machine='localhost', backend='openssh', scp_command='scp -O')
        assert transport1.scp_command == ['scp', '-O']
        assert transport1.async_backend.scp_command == ['scp', '-O']
        
        # Test complex string
        transport2 = AsyncSshTransport(machine='localhost', backend='openssh', scp_command='scp -O -v -P 2222')
        assert transport2.scp_command == ['scp', '-O', '-v', '-P', '2222']
        assert transport2.async_backend.scp_command == ['scp', '-O', '-v', '-P', '2222']
        
        # Test string with quoted arguments
        transport3 = AsyncSshTransport(machine='localhost', backend='openssh', scp_command='scp "-O" -v')
        assert transport3.scp_command == ['scp', '-O', '-v']
        assert transport3.async_backend.scp_command == ['scp', '-O', '-v']

    @pytest.mark.asyncio
    async def test_scp_command_used_in_get(self):
        """Test that custom SCP command is used in get operations."""
        custom_command = ['scp', '-O']
        transport = AsyncSshTransport(machine='localhost', backend='openssh', scp_command=custom_command)
        
        # Mock the openssh_execute method to capture the command
        original_execute = transport.async_backend.openssh_execute
        executed_commands = []
        
        async def mock_execute(commands, stdin=None, timeout=None):
            executed_commands.append(commands)
            return (0, '', '')
        
        transport.async_backend.openssh_execute = mock_execute
        
        # Try to get a file using the backend method directly (this will fail but we just want to capture the command)
        try:
            await transport.async_backend.get('/remote/file', '/local/file', False, False, False)
        except:
            pass
        
        # Verify that the custom SCP command was used correctly
        assert len(executed_commands) > 0
        command_parts = executed_commands[0]
        # The command should start with the custom command
        assert command_parts[:2] == custom_command

    @pytest.mark.asyncio
    async def test_scp_command_used_in_put(self):
        """Test that custom SCP command is used in put operations."""
        custom_command = ['scp', '-O']
        transport = AsyncSshTransport(machine='localhost', backend='openssh', scp_command=custom_command)
        
        # Mock the openssh_execute method to capture the command
        original_execute = transport.async_backend.openssh_execute
        executed_commands = []
        
        async def mock_execute(commands, stdin=None, timeout=None):
            executed_commands.append(commands)
            return (0, '', '')
        
        transport.async_backend.openssh_execute = mock_execute
        
        # Try to put a file using the backend method directly (this will fail but we just want to capture the command)
        try:
            await transport.async_backend.put('/local/file', '/remote/file', False, False, False)
        except:
            pass
        
        # Verify that the custom SCP command was used correctly
        assert len(executed_commands) > 0
        command_parts = executed_commands[0]
        # The command should start with the custom command
        assert command_parts[:2] == custom_command

    @pytest.mark.asyncio
    async def test_scp_command_used_in_copy(self):
        """Test that custom SCP command is used in copy operations."""
        custom_command = ['scp', '-O']
        transport = AsyncSshTransport(machine='localhost', backend='openssh', scp_command=custom_command)
        
        # Mock the openssh_execute method to capture the command
        original_execute = transport.async_backend.openssh_execute
        executed_commands = []
        
        async def mock_execute(commands, stdin=None, timeout=None):
            executed_commands.append(commands)
            # For path_exists and other checks, return success
            if commands[0] == 'ssh':
                return (0, '', '')
            # For scp command, return success
            elif commands[0] == 'scp':
                return (0, '', '')
            return (0, '', '')
        
        transport.async_backend.openssh_execute = mock_execute
        
        # Try to copy a file (this will fail but we just want to capture the command)
        try:
            await transport.async_backend.copy('/remote/source', '/remote/dest', False, False, False)
        except:
            pass
        
        # Verify that the custom SCP command was used correctly (it should be the last command)
        assert len(executed_commands) > 0
        # Find the SCP command in the executed commands
        scp_commands = [cmd for cmd in executed_commands if cmd[0] == 'scp']
        assert len(scp_commands) > 0, "No SCP command was executed"
        command_parts = scp_commands[0]
        # The command should start with the custom command
        assert command_parts[:2] == custom_command
