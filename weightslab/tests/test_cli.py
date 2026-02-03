"""Unit tests for weightslab CLI backend.

Tests all CLI commands to ensure they work correctly.
"""

import unittest
import json
import socket
import time
from unittest.mock import MagicMock

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from weightslab.backend.cli import (
    _handle_command,
    _sanitize_for_json,
    cli_serve,
    _server_sock
)
from weightslab.backend.ledgers import GLOBAL_LEDGER, Proxy


class TestCLISanitization(unittest.TestCase):
    """Test JSON sanitization utilities."""

    def test_sanitize_primitives(self):
        """Test that primitives pass through unchanged."""
        self.assertEqual(_sanitize_for_json(None), None)
        self.assertEqual(_sanitize_for_json(42), 42)
        self.assertEqual(_sanitize_for_json(3.14), 3.14)
        self.assertEqual(_sanitize_for_json("hello"), "hello")
        self.assertEqual(_sanitize_for_json(True), True)

    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        data = {'key': 'value', 'number': 42}
        result = _sanitize_for_json(data)
        self.assertEqual(result, {'key': 'value', 'number': 42})

    def test_sanitize_list(self):
        """Test list sanitization."""
        data = [1, 'two', 3.0, None]
        result = _sanitize_for_json(data)
        self.assertEqual(result, [1, 'two', 3.0, None])

    def test_sanitize_nested(self):
        """Test nested structure sanitization."""
        data = {
            'list': [1, 2, {'nested': 'dict'}],
            'dict': {'a': [1, 2, 3]}
        }
        result = _sanitize_for_json(data)
        self.assertEqual(result, data)

    def test_sanitize_proxy(self):
        """Test Proxy object sanitization."""
        mock_target = MagicMock()
        mock_target.value = 42

        proxy = MagicMock(spec=Proxy)
        proxy.get.return_value = mock_target

        result = _sanitize_for_json(proxy)
        # Should unwrap the proxy
        self.assertIsInstance(result, dict)


class TestCLICommands(unittest.TestCase):
    """Test all CLI command handlers."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Clear the ledger
        GLOBAL_LEDGER._models.clear()
        GLOBAL_LEDGER._dataloaders.clear()
        GLOBAL_LEDGER._optimizers.clear()
        GLOBAL_LEDGER._hyperparams.clear()

    def test_help_command(self):
        """Test the help command returns all available commands."""
        result = _handle_command('help')
        self.assertTrue(result['ok'])
        self.assertIn('commands', result)
        self.assertIn('pause / p', result['commands'])
        self.assertIn('plot_model', result['commands'])
        self.assertIn('hyperparams_examples', result)

    def test_help_alias(self):
        """Test help command aliases."""
        for alias in ['help', 'h', '?']:
            result = _handle_command(alias)
            self.assertTrue(result['ok'])
            self.assertIn('commands', result)

    def test_empty_command(self):
        """Test that empty command returns ok."""
        result = _handle_command('')
        self.assertTrue(result['ok'])
        result = _handle_command('   ')
        self.assertTrue(result['ok'])

    def test_unknown_command(self):
        """Test unknown command returns error."""
        result = _handle_command('unknown_cmd_xyz')
        self.assertFalse(result['ok'])
        self.assertIn('error', result)
        self.assertIn('unknown_command', result['error'])

    def test_status_empty_ledger(self):
        """Test status command with empty ledger."""
        result = _handle_command('status')
        self.assertTrue(result['ok'])
        self.assertIn('snapshot', result)
        self.assertEqual(result['snapshot']['models'], [])
        self.assertEqual(result['snapshot']['dataloaders'], [])
        self.assertEqual(result['snapshot']['optimizers'], [])

    def test_list_models_empty(self):
        """Test list_models with no registered models."""
        result = _handle_command('list_models')
        self.assertTrue(result['ok'])
        self.assertEqual(result['models'], [])

    def test_list_models_with_data(self):
        """Test list_models with registered model."""
        # Register a mock model
        mock_model = MagicMock()
        GLOBAL_LEDGER.register_model(mock_model, False, 'test_model')

        result = _handle_command('list_models')
        self.assertTrue(result['ok'])
        self.assertIn('test_model', result['models'])

    def test_list_dataloaders(self):
        """Test list_dataloaders command."""
        result = _handle_command('list_dataloaders')
        self.assertTrue(result['ok'])
        self.assertIn('dataloaders', result)

    def test_list_loaders_alias(self):
        """Test list_loaders alias."""
        result = _handle_command('list_loaders')
        self.assertTrue(result['ok'])
        self.assertIn('loaders', result)

        result = _handle_command('loaders')
        self.assertTrue(result['ok'])
        self.assertIn('loaders', result)

    def test_list_optimizers(self):
        """Test list_optimizers command."""
        result = _handle_command('list_optimizers')
        self.assertTrue(result['ok'])
        self.assertIn('optimizers', result)

    def test_dump_command(self):
        """Test dump command returns ledger snapshot."""
        result = _handle_command('dump')
        self.assertTrue(result['ok'])
        self.assertIn('ledger', result)
        self.assertIn('models', result['ledger'])
        self.assertIn('dataloaders', result['ledger'])
        self.assertIn('optimizers', result['ledger'])
        self.assertIn('hyperparams', result['ledger'])

    def test_plot_model_no_model(self):
        """Test plot_model when no model is registered."""
        result = _handle_command('plot_model')
        self.assertTrue(result['ok'])
        self.assertEquals(result['plot'], 'None')

    def test_plot_model_with_model(self):
        """Test plot_model with registered model."""
        # Create a mock model with __str__ method
        mock_model = MagicMock()
        mock_model.__str__ = MagicMock(return_value="Model(\n  Layer1\n  Layer2\n)")

        GLOBAL_LEDGER.register_model(mock_model, name='test_model')

        result = _handle_command('plot_model test_model')
        self.assertTrue(result['ok'])
        self.assertIn('plot', result)
        self.assertIn('model_name', result)
        self.assertEqual(result['model_name'], 'test_model')
        # Check that line breaks are preserved
        self.assertIn('\n', result['plot'])
        self.assertIn('line_count', result)
        self.assertGreater(result['line_count'], 1)

    def test_plot_model_aliases(self):
        """Test plot_model command aliases."""
        mock_model = MagicMock()
        mock_model.__str__ = MagicMock(return_value="Model()")
        GLOBAL_LEDGER.register_model(mock_model, 'test_model')

        for alias in ['plot_model', 'plot_arch', 'plot']:
            result = _handle_command(f'{alias} test_model')
            self.assertTrue(result['ok'])
            self.assertIn('plot', result)

    def test_hyperparams_list(self):
        """Test hp command lists hyperparameters."""
        result = _handle_command('hp')
        self.assertTrue(result['ok'])
        self.assertIn('hyperparams', result)

    def test_hyperparams_alias(self):
        """Test hyperparams command alias."""
        result = _handle_command('hyperparams')
        self.assertTrue(result['ok'])
        self.assertIn('hyperparams', result)

    def test_hyperparams_list_explicit(self):
        """Test hp list command."""
        for cmd in ['hp list', 'hp ls', 'hp all']:
            result = _handle_command(cmd)
            self.assertTrue(result['ok'])
            self.assertIn('hyperparams', result)

    def test_hyperparams_show(self):
        """Test showing specific hyperparameter."""
        # Register a mock hyperparameter
        GLOBAL_LEDGER.register_hyperparams({'key': 'value'}, name='test_hp')

        result = _handle_command('hp test_hp')
        self.assertTrue(result['ok'])
        self.assertEqual(result['name'], 'test_hp')
        self.assertIn('hyperparams', result)

    def test_list_uids_no_loader(self):
        """Test list_uids with no dataloaders."""
        result = _handle_command('list_uids')
        self.assertTrue(result['ok'])
        self.assertIn('uids', result)
        self.assertEqual(result['uids'], {})

    def test_discard_no_uid(self):
        """Test discard command without UIDs."""
        result = _handle_command('discard')
        self.assertFalse(result['ok'])
        self.assertIn('error', result)
        self.assertIn('usage', result['error'])

    def test_undiscard_no_uid(self):
        """Test undiscard command without UIDs."""
        result = _handle_command('undiscard')
        self.assertFalse(result['ok'])
        self.assertIn('error', result)
        self.assertIn('usage', result['error'])

    def test_add_tag_insufficient_args(self):
        """Test add_tag with insufficient arguments."""
        result = _handle_command('add_tag')
        self.assertFalse(result['ok'])
        self.assertIn('usage', result['error'])

        result = _handle_command('add_tag uid1')
        self.assertFalse(result['ok'])
        self.assertIn('usage', result['error'])

    def test_set_hp_insufficient_args(self):
        """Test set_hp with insufficient arguments."""
        result = _handle_command('set_hp')
        self.assertFalse(result['ok'])
        self.assertIn('usage', result['error'])

    def test_operate_insufficient_args(self):
        """Test operate command with insufficient arguments."""
        result = _handle_command('operate 1 2')
        self.assertFalse(result['ok'])
        self.assertIn('usage', result['error'])


class TestCLIServer(unittest.TestCase):
    """Test CLI server functionality."""

    def tearDown(self):
        """Clean up after tests."""
        global _server_thread, _server_sock
        if _server_sock:
            try:
                _server_sock.close()
            except Exception:
                pass
            _server_sock = None
        _server_thread = None

    def test_cli_serve_starts(self):
        """Test that CLI server starts successfully."""
        result = cli_serve(cli_host='127.0.0.1', cli_port=0, spawn_client=False)

        self.assertTrue(result['ok'])
        self.assertIn('host', result)
        self.assertIn('port', result)
        self.assertGreater(result['port'], 0)

        # Give server time to start
        time.sleep(0.1)

        # Test connection
        try:
            sock = socket.create_connection((result['host'], result['port']), timeout=2)
            sock.close()
        except Exception as e:
            self.fail(f"Could not connect to server: {e}")

    def test_cli_serve_port_binding(self):
        """Test server binds to specified port."""
        # Use port 0 to let OS assign
        result = cli_serve(cli_host='127.0.0.1', cli_port=0, spawn_client=False)
        self.assertTrue(result['ok'])
        self.assertGreater(result['port'], 0)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI server-client communication."""

    @classmethod
    def setUpClass(cls):
        """Start CLI server for integration tests."""
        cls.server_info = cli_serve(cli_host='127.0.0.1', cli_port=0, spawn_client=False)
        if not cls.server_info['ok']:
            raise RuntimeError("Failed to start CLI server for integration tests")
        time.sleep(0.2)  # Give server time to fully start

    @classmethod
    def tearDownClass(cls):
        """Stop CLI server after integration tests."""
        global _server_sock
        if _server_sock:
            try:
                _server_sock.close()
            except Exception:
                pass

    def _send_command(self, cmd: str) -> dict:
        """Helper to send command to server and get response."""
        sock = socket.create_connection(
            (self.server_info['host'], self.server_info['port']),
            timeout=5
        )
        f = sock.makefile('rwb')

        # Send command
        f.write((cmd + '\n').encode('utf8'))
        f.flush()

        # Read response
        response_line = f.readline()
        response = json.loads(response_line.decode('utf8'))

        f.close()
        sock.close()

        return response

    def test_integration_help(self):
        """Test help command through server."""
        response = self._send_command('help')
        self.assertTrue(response['ok'])
        self.assertIn('commands', response)

    def test_integration_status(self):
        """Test status command through server."""
        response = self._send_command('status')
        self.assertTrue(response['ok'])
        self.assertIn('snapshot', response)

    def test_integration_list_models(self):
        """Test list_models through server."""
        response = self._send_command('list_models')
        self.assertTrue(response['ok'])
        self.assertIn('models', response)

    def test_integration_unknown_command(self):
        """Test unknown command through server."""
        response = self._send_command('invalid_command_xyz')
        self.assertFalse(response['ok'])
        self.assertIn('error', response)

    def test_integration_quit(self):
        """Test quit command closes connection."""
        sock = socket.create_connection(
            (self.server_info['host'], self.server_info['port']),
            timeout=5
        )
        f = sock.makefile('rwb')

        # Send quit
        f.write(b'quit\n')
        f.flush()

        # Read goodbye
        response = json.loads(f.readline().decode('utf8'))
        self.assertTrue(response['ok'])
        self.assertTrue(response.get('bye'))

        f.close()
        sock.close()


def run_tests():
    """Run all CLI tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCLISanitization))
    suite.addTests(loader.loadTestsFromTestCase(TestCLICommands))
    suite.addTests(loader.loadTestsFromTestCase(TestCLIServer))
    suite.addTests(loader.loadTestsFromTestCase(TestCLIIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
