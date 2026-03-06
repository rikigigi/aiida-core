import asyncio
import os
import stat
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from aiida.transports.plugins.async_backend import _OpenSSH, get_openssh_version, is_openssh_9_or_higher
from aiida.transports.plugins.ssh_async import AsyncSshTransport


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

        with async_transport as transport:
            # Each operation should fail but release the semaphore
            with pytest.raises(OSError, match='Error while downloading file'):
                await transport.getfile_async('non_existing', local_dir)

            with pytest.raises(OSError, match='Error while uploading file'):
                await transport.puttree_async(local_dir, 'target_dir')

            # This would deadlock if semaphore wasn't released after previous errors
            with pytest.raises(OSError, match='Error while downloading file'):
                await transport.getfile_async('non_existing', local_dir)

        assert transport._semaphore._value == 1, 'Semaphore should be fully released'

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
    """Test that get_openssh_version() parses version correctly and caches results."""
    # Clear the cache before testing
    get_openssh_version.cache_clear()

    mock_result = MagicMock()
    mock_result.stderr = 'OpenSSH_9.0p1, OpenSSL 3.0.2'
    mock_result.stdout = ''

    with patch('aiida.transports.plugins.async_backend.subprocess.run', return_value=mock_result):
        assert get_openssh_version() == 9

    # Clear cache to test other version
    get_openssh_version.cache_clear()

    mock_result.stderr = 'OpenSSH_8.9p1, OpenSSL 3.0.2'

    with patch('aiida.transports.plugins.async_backend.subprocess.run', return_value=mock_result):
        assert get_openssh_version() == 8

    # Clear cache after test
    get_openssh_version.cache_clear()


def test_is_openssh_9_or_higher():
    """Test that is_openssh_9_or_higher() correctly identifies version thresholds."""
    # Clear the cache before testing
    get_openssh_version.cache_clear()

    mock_result = MagicMock()
    mock_result.stderr = 'OpenSSH_9.0p1, OpenSSL 3.0.2'
    mock_result.stdout = ''

    with patch('aiida.transports.plugins.async_backend.subprocess.run', return_value=mock_result):
        assert is_openssh_9_or_higher() is True

    # Clear cache to test older version
    get_openssh_version.cache_clear()

    mock_result.stderr = 'OpenSSH_8.9p1, OpenSSL 3.0.2'

    with patch('aiida.transports.plugins.async_backend.subprocess.run', return_value=mock_result):
        assert is_openssh_9_or_higher() is False

    # Clear cache after test
    get_openssh_version.cache_clear()


def test_openssh_version_caching():
    """Test that get_openssh_version() caches its result."""
    get_openssh_version.cache_clear()

    mock_result = MagicMock()
    mock_result.stderr = 'OpenSSH_9.0p1, OpenSSL 3.0.2'
    mock_result.stdout = ''

    with patch('aiida.transports.plugins.async_backend.subprocess.run', return_value=mock_result) as mock_run:
        # First call should invoke subprocess.run
        version1 = get_openssh_version()
        assert version1 == 9
        assert mock_run.call_count == 1

        # Second call should use cached value
        version2 = get_openssh_version()
        assert version2 == 9
        assert mock_run.call_count == 1  # Still 1, not 2

    get_openssh_version.cache_clear()


class TestSshCommandGenerator:
    """Tests for ssh_command_generator escaping."""

    def test_escapes_shell_chars_and_paths(self):
        """Test $, `, " escaped and paths safely quoted."""

        class TestOpenSSH(_OpenSSH):
            def __init__(self):
                self.machine = 'localhost'
                self.bash_command = 'bash -c '

        openssh = TestOpenSSH()

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


class TestEscapeForGlob:
    """Tests for glob path escaping that preserves wildcards while escaping dangerous chars."""

    @pytest.fixture
    def openssh_backend(self):
        class TestOpenSSH(_OpenSSH):
            def __init__(self):
                self.machine = 'localhost'

        return TestOpenSSH()

    def test_preserves_wildcards_escapes_dangerous_chars(self, openssh_backend):
        """Test wildcards (*, ?, []) preserved while dangerous chars escaped."""
        # Wildcards preserved, dangerous chars escaped
        assert (
            openssh_backend._escape_for_glob('/home/$USER/my files/[0-9]*.log') == '/home/\\$USER/my\\ files/[0-9]*.log'
        )
        # Command injection attempt escaped
        assert openssh_backend._escape_for_glob('/path;rm -rf /*') == '/path\\;rm\\ -rf\\ /*'


class TestScpEscaping:
    """Tests for scp path escaping for OpenSSH 9+ (SFTP) vs <9 (RCP)."""

    @pytest.fixture
    def openssh_backend(self):
        class TestOpenSSH(_OpenSSH):
            def __init__(self):
                self.machine = 'localhost'
                self.scp_command = ['scp']

        return TestOpenSSH()

    def test_escape_for_rcp(self, openssh_backend):
        """Test RCP mode escapes shell metacharacters."""
        assert openssh_backend._escape_for_rcp('/path/with spaces/$VAR;cmd') == '/path/with\\ spaces/\\$VAR\\;cmd'

    def test_escape_for_scp_version_aware(self, openssh_backend):
        """Test _escape_for_scp behavior differs by OpenSSH version."""
        get_openssh_version.cache_clear()
        path = '/path/with spaces'

        with patch('aiida.transports.plugins.async_backend.is_openssh_9_or_higher', return_value=True):
            assert openssh_backend._escape_for_scp(path) == path  # SFTP: no escaping

        with patch('aiida.transports.plugins.async_backend.is_openssh_9_or_higher', return_value=False):
            assert openssh_backend._escape_for_scp(path) == '/path/with\\ spaces'  # RCP: escaped

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


class TestScpIntegration:
    """Integration tests for scp with special characters in filenames."""

    @pytest.fixture
    def test_files(self, tmp_path):
        remote_dir = tmp_path / 'remote'
        local_dir = tmp_path / 'local'
        remote_dir.mkdir()
        local_dir.mkdir()
        special_files = ['file with spaces.txt', "file'quote.txt", 'file$dollar.txt', 'aiida.pdos_atm#2(Al)_wfc#2(p).txt']
        for f in special_files:
            (remote_dir / f).write_text(f'content of {f}')
        return {'remote': remote_dir, 'local': local_dir, 'files': special_files}

    @pytest.fixture
    def openssh_backend(self):
        class TestOpenSSH(_OpenSSH):
            def __init__(self):
                self.machine = 'localhost'

        return TestOpenSSH()

    def test_scp_with_special_chars(self, test_files, openssh_backend):
        """Test scp in both SFTP and RCP modes with special characters."""
        for filename in test_files['files']:
            source = test_files['remote'] / filename

            # SFTP mode (default on OpenSSH 9+)
            dest_sftp = test_files['local'] / f'sftp_{filename}'
            result = subprocess.run(['scp', f'localhost:{source}', str(dest_sftp)], capture_output=True, check=False)
            assert result.returncode == 0 and dest_sftp.read_text() == f'content of {filename}'

            # RCP mode (forced with -O)
            dest_rcp = test_files['local'] / f'rcp_{filename}'
            escaped = openssh_backend._escape_for_rcp(str(source))
            result = subprocess.run(
                ['scp', '-O', f'localhost:{escaped}', str(dest_rcp)], capture_output=True, check=False
            )
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
        executed_commands = []
        
        async def mock_execute(commands, stdin=None, timeout=None):
            executed_commands.append(commands)
            # For path_exists and other checks, return appropriate responses
            if commands[0] == 'ssh':
                command_str = ' '.join(commands)
                
                # For glob operations, return fake file data
                if 'find' in command_str and '-print0' in command_str:
                    return (0, '/scratch2/a/./out/file1.txt\0/scratch2/a/./out/file2.txt', '')
                
                # For path existence checks
                elif 'test -e' in command_str:
                    return (0, '', '')  # Path exists
                elif 'test -f' in command_str:
                    return (1, '', 'Not a file')  # Not a regular file (it's a directory)
                elif 'test -d' in command_str:
                    return (0, '', '')  # It's a directory
                
                # For ls commands
                elif 'ls' in command_str:
                    return (0, '', '')  # Empty directory listing
                
                return (0, '', '')
            # For scp command, return success
            elif commands[0] == 'scp':
                return (0, '', '')
            return (0, '', '')

        transport.async_backend.openssh_execute = mock_execute

        # Try to copy a file (this will fail but we just want to capture the command)
        try:
            await transport.async_backend.copy('/scratch2/a/./out/*', '/scratch2/b/out', False, False, False)
        except:
            pass
        
        print(executed_commands)

        # Verify that the SSH command was used instead of SCP
        assert len(executed_commands) > 0
        # Find the SSH command in the executed commands
        ssh_commands = [cmd for cmd in executed_commands if cmd[0] == 'ssh']
        assert len(ssh_commands) > 0, "No SSH command was executed"
        command_parts = ssh_commands[-1]
        # The command should contain 'cp -rL'
        assert 'cp -rL' in ' '.join(command_parts)


class TestSecurityEscaping:
    """Security-focused tests for command and path escaping."""

    @pytest.fixture
    def openssh_backend(self):
        class TestOpenSSH(_OpenSSH):
            def __init__(self):
                self.machine = 'localhost'
                self.scp_command = ['scp']

        return TestOpenSSH()

    def test_escape_command_injection_attempts(self, openssh_backend):
        """Test that command injection attempts in paths are properly escaped."""
        malicious_paths = [
            '/path; rm -rf /',           # Command injection with semicolon
            r'/path $(whoami)',           # Command substitution
            r'/path `whoami`',            # Backtick command substitution
            '/path && echo hacked',      # Command chaining
            '/path | cat /etc/passwd',   # Pipe to command
            '/path > /tmp/hack',         # Output redirection
            '/path < /etc/passwd',       # Input redirection
            r'/path $(rm /tmp/*)',        # Nested command substitution
            r'/path $HOME',               # Environment variable expansion
            r'/path ${PATH}',             # Braced environment variable
        ]
        
        for path in malicious_paths:
            escaped = openssh_backend._escape_for_glob(path)
            
            # Check that dangerous characters are escaped with backslashes
            # Count exact number of escapes to ensure all instances are escaped
            if ';' in path:
                original_count = path.count(';')
                escaped_count = escaped.count(r'\;')
                assert escaped_count == original_count, f"Semicolon not fully escaped in {path}: expected {original_count}, got {escaped_count}"
            if '$' in path:
                original_count = path.count('$')
                escaped_count = escaped.count(r'\$')
                assert escaped_count == original_count, f"Dollar not fully escaped in {path}: expected {original_count}, got {escaped_count}"
            if '`' in path:
                original_count = path.count('`')
                escaped_count = escaped.count(r'\`')
                assert escaped_count == original_count, f"Backtick not fully escaped in {path}: expected {original_count}, got {escaped_count}"
            if '|' in path:
                original_count = path.count('|')
                escaped_count = escaped.count(r'\|')
                assert escaped_count == original_count, f"Pipe not fully escaped in {path}: expected {original_count}, got {escaped_count}"
            if '>' in path:
                original_count = path.count('>')
                escaped_count = escaped.count(r'\>')
                assert escaped_count == original_count, f"Redirection not fully escaped in {path}: expected {original_count}, got {escaped_count}"
            if '<' in path:
                original_count = path.count('<')
                escaped_count = escaped.count(r'\<')
                assert escaped_count == original_count, f"Redirection not fully escaped in {path}: expected {original_count}, got {escaped_count}"
            if '&' in path:
                original_count = path.count('&')
                escaped_count = escaped.count(r'\&')
                assert escaped_count == original_count, f"Ampersand not fully escaped in {path}: expected {original_count}, got {escaped_count}"

    def test_escape_unicode_and_special_chars(self, openssh_backend):
        """Test escaping of Unicode characters and special filesystem characters."""
        special_paths = [
            '/path/with/unicode/ñáéíóú',  # Unicode characters
            '/path/with/spaces and spaces', # Multiple spaces
            '/path/with\ttabs',           # Tab characters
            '/path/with\nnewlines',       # Newline characters
            '/path/with[brackets]',       # Square brackets
            '/path/with{braces}',         # Curly braces
            '/path/with(parens)',         # Parentheses
            '/path/with@symbols',         # At symbol
            '/path/with#hash',            # Hash symbol
            '/path/with!exclamation',     # Exclamation mark
        ]
        
        for path in special_paths:
            escaped = openssh_backend._escape_for_glob(path)
            # Should not raise exceptions and should escape appropriately
            assert escaped is not None
            assert len(escaped) > 0

    def test_glob_behavior_with_wildcards(self, openssh_backend):
        """Test that glob patterns preserve wildcards but escape dangerous characters."""
        test_cases = [
            ('/path/*.txt', '/path/*.txt'),        # Simple wildcard preserved
            ('/path/file[1-9].dat', '/path/file[1-9].dat'), # Character class preserved
            ('/path/**/*.log', '/path/**/*.log'), # Recursive wildcard preserved
            ('/path; rm -rf /', r'/path\;\ rm\ -rf\ /'), # Dangerous pattern escaped
            (r'/path$(cmd)', r'/path\$\(cmd\)'), # Command substitution escaped
            (r'/path`cmd`', r'/path\`cmd\`'), # Backtick substitution escaped
            ('/path with spaces/*.txt', r'/path\ with\ spaces/*.txt'), # Spaces escaped, wildcard preserved
        ]
        
        for pattern, expected_escaped in test_cases:
            escaped = openssh_backend._escape_for_glob(pattern)
            assert escaped == expected_escaped, f"Expected {expected_escaped}, got {escaped} for {pattern}"

    def test_escape_path_traversal_attempts(self, openssh_backend):
        """Test that path traversal attempts are handled safely."""
        traversal_paths = [
            '/path/../etc/passwd',       # Simple traversal
            '/path/../../../etc/passwd', # Multiple traversals
            '/path/./current/file',       # Current directory references
            '/path/subdir/../file',      # Mixed traversal
            '/path/../',                  # Traversal at end
            '../relative/path',          # Relative traversal
        ]
        
        for path in traversal_paths:
            escaped = openssh_backend._escape_for_glob(path)
            # Should escape the path but not necessarily prevent traversal
            # (traversal prevention is filesystem's responsibility)
            assert escaped is not None
            assert len(escaped) > 0

    def test_escape_environment_variable_injection(self, openssh_backend):
        """Test escaping of environment variable injection attempts."""
        env_var_paths = [
            r'/path/$USER/file',          # Simple variable (raw string)
            r'/path/${HOME}/file',        # Braced variable (raw string)
            r'/path/$PATH:/etc',          # Path variable (raw string)
            r'/path/${HOSTNAME}',         # System variable (raw string)
            r'/path/$((1+1))',            # Arithmetic expansion (raw string)
            r'/path/${VAR:-default}',     # Variable with default (raw string)
        ]
        
        for path in env_var_paths:
            escaped = openssh_backend._escape_for_glob(path)
            # Environment variables should be escaped with backslashes
            # Count exact number of dollar signs and ensure all are escaped
            original_dollars = path.count('$')
            escaped_dollars = escaped.count(r'\$')
            assert escaped_dollars == original_dollars, f"Dollar signs not fully escaped in {path}: expected {original_dollars}, got {escaped_dollars}"

    def test_escape_symlink_race_conditions(self, openssh_backend):
        """Test that symlink-related operations are safely escaped."""
        # While escaping can't prevent symlink races, it should not make them worse
        symlink_paths = [
            '/path/to/symlink',
            '/path/with spaces/symlink',
            r'/path/with$var/symlink',
            '/path/with;cmd/symlink',
        ]
        
        for path in symlink_paths:
            escaped = openssh_backend._escape_for_glob(path)
            assert escaped is not None
            assert len(escaped) > 0
            # Dangerous characters should still be escaped - count exactly
            if ';' in path:
                original_count = path.count(';')
                escaped_count = escaped.count(r'\;')
                assert escaped_count == original_count, f"Semicolons not fully escaped: expected {original_count}, got {escaped_count}"
            if '$' in path:
                original_count = path.count('$')
                escaped_count = escaped.count(r'\$')
                assert escaped_count == original_count, f"Dollars not fully escaped: expected {original_count}, got {escaped_count}"

    def test_escape_extremely_long_paths(self, openssh_backend):
        """Test escaping of extremely long paths that might cause issues."""
        long_path = '/very/' + 'long/' * 100 + 'path'
        escaped = openssh_backend._escape_for_glob(long_path)
        
        # Should handle long paths without crashing
        assert escaped is not None
        # The escaped version should be at least as long as original
        assert len(escaped) >= len(long_path)
        
        # Test path that's exactly at common limits (e.g., 255, 4096 chars)
        for length in [255, 4096]:
            test_path = '/test/' + 'a' * length
            escaped = openssh_backend._escape_for_glob(test_path)
            assert escaped is not None
            # Escaped version should be >= original length
            assert len(escaped) >= len(test_path)

    def test_escape_mixed_quoting_attempts(self, openssh_backend):
        """Test escaping of paths with mixed quoting attempts."""
        mixed_quote_paths = [
            r'/path/with"single"quotes',  # Single quotes (raw string)
            r'/path/with"double"quotes',  # Double quotes (raw string)
            r'/path/with\"escaped\"quotes', # Escaped quotes (raw string)
            r"'/path/starting/with/quote",   # Starting with quote (raw string)
            r"/path/ending/with/quote'",    # Ending with quote (raw string)
            r'/path/with\backslashes',    # Backslashes (raw string)
        ]
        
        for path in mixed_quote_paths:
            escaped = openssh_backend._escape_for_glob(path)
            assert escaped is not None
            # Quotes should be escaped with backslashes
            if "'" in path:
                assert "\\'" in escaped or escaped.count("'") > path.count("'")
            if '"' in path:
                assert '\\"' in escaped or escaped.count('"') > path.count('"')
            if '\\' in path:
                assert escaped.count('\\') >= path.count('\\')

    def test_escape_null_bytes_and_control_chars(self, openssh_backend):
        """Test escaping of null bytes and control characters."""
        control_char_paths = [
            '/path/with\x00null',        # Null byte
            '/path/with\x01control',     # Control-A
            '/path/with\x1fcontrol',     # Control-_
            '/path/with\t\n\rwhitespace', # Various whitespace
        ]
        
        for path in control_char_paths:
            escaped = openssh_backend._escape_for_glob(path)
            assert escaped is not None
            # Should handle control characters safely
            assert len(escaped) > 0

    def test_escape_glob_negation_patterns(self, openssh_backend):
        """Test escaping of glob negation patterns that could be dangerous."""
        negation_patterns = [
            ('/path/[!a-z]*.txt', '/path/[\!a-z]*.txt'),          # Negated character class
            ('/path/*[!0-9]', '/path/*[\!0-9]'),             # Negation at end
            ('/path/[!abc]file', '/path/[\!abc]file'),           # Simple negation
        ]
        
        for pattern, expected_escaped in negation_patterns:
            escaped = openssh_backend._escape_for_glob(pattern)
            assert escaped is not None
            # The negation pattern should be escaped (exclamation mark escaped)
            assert escaped == expected_escaped, f"Expected {expected_escaped}, got {escaped}"

    def test_escape_extended_glob_patterns(self, openssh_backend):
        """Test escaping of extended glob patterns."""
        extended_patterns = [
            ('/path/@(file1|file2).txt', '/path/@\(file1\|file2\).txt'),   # Alternation
            ('/path/+(file1|file2).txt', '/path/+\(file1\|file2\).txt'),   # One or more
            ('/path/*(file1|file2).txt', '/path/*\(file1\|file2\).txt'),   # Zero or more
            ('/path/?(file1|file2).txt', '/path/?\(file1\|file2\).txt'),   # Zero or one
            ('/path/!(file1|file2).txt', '/path/\!\(file1\|file2\).txt'),   # Negation
        ]
        
        for pattern, expected_escaped in extended_patterns:
            escaped = openssh_backend._escape_for_glob(pattern)
            assert escaped is not None
            # Extended glob patterns should be escaped (special chars escaped)
            assert escaped == expected_escaped, f"Expected {expected_escaped}, got {escaped}"

    def test_escape_shell_metacharacters_comprehensive(self, openssh_backend):
        """Comprehensive test of all shell metacharacters."""
        metacharacters = '; & | < > ( ) $ ` \\ " \' ! # [ ] { } * ? ~'
        
        for char in metacharacters:
            path = f'/path/with{char}char'
            escaped = openssh_backend._escape_for_glob(path)
            
            # Most metacharacters should be escaped (prefixed with backslash)
            if char in [';', '&', '|', '<', '>', '$', '`', '!', '#']:
                assert f'\\{char}' in escaped, f"Metacharacter {char} not escaped"
            
            # Some characters are preserved for glob functionality but may also be escaped
            elif char in ['*', '?', '[', ']', '{', '}', '~']:
                # These may be preserved or escaped depending on context
                pass
            
            # Quotes and backslashes should be escaped
            elif char in ['"', "'", '\\']:
                assert f'\\{char}' in escaped, f"Quote/backslash {char} not escaped"

    def test_escape_real_world_malicious_filenames(self, openssh_backend):
        """Test escaping of real-world malicious filename examples."""
        real_world_malicious = [
            # Common attack patterns
            '/etc/passwd',
            '/etc/shadow',
            '/home/user/.ssh/authorized_keys',
            '/home/user/.bash_history',
            '/proc/self/environ',
            
            # Web-related attacks
            '/var/www/html/index.php',
            '/var/www/.htaccess',
            
            # Command injection via filenames (use raw strings)
            r'$(id)',
            r'`whoami`',
            '; id',
            '&& cat /etc/passwd',
            
            # Path traversal
            '../../../etc/passwd',
            '/var/www/../../etc/passwd',
            
            # Mixed attacks (use raw strings)
            r'/tmp/$(whoami)_`id`.txt',
            '/home/user/.ssh/authorized_keys; echo hacked',
        ]
        
        for filename in real_world_malicious:
            # Test both as full path and as part of path
            for path in [filename, f'/base/dir/{filename}']:
                escaped = openssh_backend._escape_for_glob(path)
                assert escaped is not None
                assert len(escaped) > 0
                
                # Check that common dangerous patterns are escaped with backslashes - count exactly
                if '$(' in path:
                    original_dollars = path.count('$')
                    escaped_dollars = escaped.count(r'\$')
                    assert escaped_dollars == original_dollars, f"Dollars not fully escaped in {path}"
                if '`' in path:
                    original_backticks = path.count('`')
                    escaped_backticks = escaped.count(r'\`')
                    assert escaped_backticks == original_backticks, f"Backticks not fully escaped in {path}"
                if ';' in path:
                    original_semicolons = path.count(';')
                    escaped_semicolons = escaped.count(r'\;')
                    assert escaped_semicolons == original_semicolons, f"Semicolons not fully escaped in {path}"
                if '&&' in path:
                    original_ampersands = path.count('&')
                    escaped_ampersands = escaped.count(r'\&')
                    assert escaped_ampersands == original_ampersands, f"Ampersands not fully escaped in {path}"