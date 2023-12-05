"""Tests for the hooks feature."""

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rx.hooks import (
    HOOK_TIMEOUT_SECONDS,
    HookConfig,
    call_hook_sync,
    generate_request_id,
    get_effective_hooks,
    get_hook_env_config,
)
from rx.models import FileScannedPayload, MatchFoundPayload, TraceCompletePayload
from rx.request_store import (
    RequestInfo,
    _lock,
    _requests,
    clear_old_requests,
    get_request,
    get_store_stats,
    increment_hook_counter,
    list_requests,
    store_request,
    update_request,
)
from rx.trace import HookCallbacks


class TestHookConfig:
    """Tests for HookConfig dataclass."""

    def test_has_any_hook_none(self):
        """Test has_any_hook returns False when no hooks configured."""
        config = HookConfig()
        assert config.has_any_hook() is False

    def test_has_any_hook_with_file_hook(self):
        """Test has_any_hook returns True when on_file_url is set."""
        config = HookConfig(on_file_url="http://example.com/hook")
        assert config.has_any_hook() is True

    def test_has_any_hook_with_match_hook(self):
        """Test has_any_hook returns True when on_match_url is set."""
        config = HookConfig(on_match_url="http://example.com/hook")
        assert config.has_any_hook() is True

    def test_has_any_hook_with_complete_hook(self):
        """Test has_any_hook returns True when on_complete_url is set."""
        config = HookConfig(on_complete_url="http://example.com/hook")
        assert config.has_any_hook() is True

    def test_has_match_hook_false(self):
        """Test has_match_hook returns False when not configured."""
        config = HookConfig(on_file_url="http://example.com/hook")
        assert config.has_match_hook() is False

    def test_has_match_hook_true(self):
        """Test has_match_hook returns True when on_match_url is set."""
        config = HookConfig(on_match_url="http://example.com/hook")
        assert config.has_match_hook() is True


class TestGetEffectiveHooks:
    """Tests for get_effective_hooks function."""

    def test_no_hooks_configured(self):
        """Test returns empty config when no hooks configured."""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to pick up cleared env
            import importlib

            from rx import hooks

            importlib.reload(hooks)

            config = hooks.get_effective_hooks()
            assert config.on_file_url is None
            assert config.on_match_url is None
            assert config.on_complete_url is None

    def test_custom_hooks_override_env(self):
        """Test custom hooks override env vars."""
        config = get_effective_hooks(
            custom_on_file="http://custom.com/file",
            custom_on_match="http://custom.com/match",
            custom_on_complete="http://custom.com/complete",
        )
        assert config.on_file_url == "http://custom.com/file"
        assert config.on_match_url == "http://custom.com/match"
        assert config.on_complete_url == "http://custom.com/complete"

    def test_disable_custom_hooks(self):
        """Test RX_DISABLE_CUSTOM_HOOKS ignores custom hooks."""
        with patch.dict(os.environ, {'RX_DISABLE_CUSTOM_HOOKS': 'true'}, clear=False):
            import importlib

            from rx import hooks

            importlib.reload(hooks)

            config = hooks.get_effective_hooks(
                custom_on_file="http://custom.com/file",
            )
            # Custom hooks should be ignored when disabled
            # Would return env var value (None since not set)
            assert config.on_file_url is None


class TestGenerateRequestId:
    """Tests for generate_request_id function."""

    def test_generates_uuid_string(self):
        """Test generates a valid UUID string."""
        request_id = generate_request_id()
        assert isinstance(request_id, str)
        assert len(request_id) == 36  # UUID format: 8-4-4-4-12

    def test_generates_unique_ids(self):
        """Test generates unique IDs on each call."""
        ids = [generate_request_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique


class TestBuildPayloads:
    """Tests for payload Pydantic models."""

    def test_build_file_scanned_payload(self):
        """Test building file scanned payload."""
        payload_model = FileScannedPayload(
            request_id="test-123",
            file_path="/var/log/app.log",
            file_size_bytes=1024,
            scan_time_ms=100,
            matches_count=5,
        )
        payload = payload_model.model_dump()
        assert payload['event'] == 'file_scanned'
        assert payload['request_id'] == 'test-123'
        assert payload['file_path'] == '/var/log/app.log'
        assert payload['file_size_bytes'] == 1024
        assert payload['scan_time_ms'] == 100
        assert payload['matches_count'] == 5

    def test_build_match_found_payload(self):
        """Test building match found payload."""
        payload_model = MatchFoundPayload(
            request_id="test-123",
            file_path="/var/log/app.log",
            pattern="error.*",
            offset=500,
            line_number=42,
        )
        payload = payload_model.model_dump()
        assert payload['event'] == 'match_found'
        assert payload['request_id'] == 'test-123'
        assert payload['file_path'] == '/var/log/app.log'
        assert payload['pattern'] == 'error.*'
        assert payload['offset'] == 500
        assert payload['line_number'] == 42

    def test_build_match_found_payload_without_line_number(self):
        """Test building match found payload without line number."""
        payload_model = MatchFoundPayload(
            request_id="test-123",
            file_path="/var/log/app.log",
            pattern="error.*",
            offset=500,
        )
        payload = payload_model.model_dump()
        assert payload['line_number'] is None

    def test_build_trace_complete_payload(self):
        """Test building trace complete payload."""
        payload_model = TraceCompletePayload(
            request_id="test-123",
            paths='/var/log/app.log,/var/log/error.log',
            patterns='error.*,warning.*',
            total_files_scanned=2,
            total_files_skipped=1,
            total_matches=10,
            total_time_ms=500,
        )
        payload = payload_model.model_dump()
        assert payload['event'] == 'trace_complete'
        assert payload['request_id'] == 'test-123'
        assert payload['paths'] == '/var/log/app.log,/var/log/error.log'
        assert payload['patterns'] == 'error.*,warning.*'
        assert payload['total_files_scanned'] == 2
        assert payload['total_files_skipped'] == 1
        assert payload['total_matches'] == 10
        assert payload['total_time_ms'] == 500


class TestCallHookSync:
    """Tests for synchronous hook calls."""

    @patch('rx.hooks.httpx.Client')
    def test_successful_hook_call(self, mock_client_class):
        """Test successful hook call returns True."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = call_hook_sync(
            "http://example.com/hook",
            {"key": "value"},
            "on_complete",
        )
        assert result is True
        mock_client.get.assert_called_once()

    @patch('rx.hooks.httpx.Client')
    def test_failed_hook_call(self, mock_client_class):
        """Test failed hook call returns False."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = call_hook_sync(
            "http://example.com/hook",
            {"key": "value"},
            "on_complete",
        )
        assert result is False

    @patch('rx.hooks.httpx.Client')
    def test_hook_call_exception(self, mock_client_class):
        """Test hook call exception returns False."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_client

        result = call_hook_sync(
            "http://example.com/hook",
            {"key": "value"},
            "on_complete",
        )
        assert result is False


class TestRequestStore:
    """Tests for request store functionality."""

    def setup_method(self):
        """Clear request store before each test."""
        with _lock:
            _requests.clear()

    def test_store_and_get_request(self):
        """Test storing and retrieving a request."""
        info = RequestInfo(
            request_id="test-123",
            paths=["/var/log/app.log"],
            patterns=["error.*"],
            max_results=100,
            started_at=datetime.now(),
        )
        store_request(info)

        retrieved = get_request("test-123")
        assert retrieved is not None
        assert retrieved.request_id == "test-123"
        assert retrieved.paths == ["/var/log/app.log"]

    def test_get_nonexistent_request(self):
        """Test getting a nonexistent request returns None."""
        result = get_request("nonexistent")
        assert result is None

    def test_update_request(self):
        """Test updating a request."""
        info = RequestInfo(
            request_id="test-123",
            paths=["/var/log/app.log"],
            patterns=["error.*"],
            max_results=100,
            started_at=datetime.now(),
        )
        store_request(info)

        update_request("test-123", total_matches=50, total_files_scanned=5)

        retrieved = get_request("test-123")
        assert retrieved.total_matches == 50
        assert retrieved.total_files_scanned == 5

    def test_update_nonexistent_request(self):
        """Test updating nonexistent request returns False."""
        result = update_request("nonexistent", total_matches=50)
        assert result is False

    def test_increment_hook_counter(self):
        """Test incrementing hook counters."""
        info = RequestInfo(
            request_id="test-123",
            paths=["/var/log/app.log"],
            patterns=["error.*"],
            max_results=100,
            started_at=datetime.now(),
        )
        store_request(info)

        increment_hook_counter("test-123", "on_file", True)
        increment_hook_counter("test-123", "on_file", False)
        increment_hook_counter("test-123", "on_match", True)
        increment_hook_counter("test-123", "on_complete", True)

        retrieved = get_request("test-123")
        assert retrieved.hook_on_file_success == 1
        assert retrieved.hook_on_file_failed == 1
        assert retrieved.hook_on_match_success == 1
        assert retrieved.hook_on_complete_success == 1

    def test_list_requests(self):
        """Test listing requests."""
        for i in range(5):
            info = RequestInfo(
                request_id=f"test-{i}",
                paths=["/var/log/app.log"],
                patterns=["error.*"],
                max_results=100,
                started_at=datetime.now(),
            )
            store_request(info)

        requests = list_requests(limit=3)
        assert len(requests) == 3

    def test_clear_old_requests(self):
        """Test clearing old completed requests."""
        old_time = datetime(2020, 1, 1)
        info = RequestInfo(
            request_id="old-request",
            paths=["/var/log/app.log"],
            patterns=["error.*"],
            max_results=100,
            started_at=old_time,
            completed_at=old_time,
        )
        store_request(info)

        # Clear requests older than 1 second
        cleared = clear_old_requests(max_age_seconds=1)
        assert cleared == 1
        assert get_request("old-request") is None

    def test_get_store_stats(self):
        """Test getting store statistics."""
        info1 = RequestInfo(
            request_id="test-1",
            paths=["/var/log/app.log"],
            patterns=["error.*"],
            max_results=100,
            started_at=datetime.now(),
        )
        info2 = RequestInfo(
            request_id="test-2",
            paths=["/var/log/app.log"],
            patterns=["error.*"],
            max_results=100,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        store_request(info1)
        store_request(info2)

        stats = get_store_stats()
        assert stats['total_requests'] == 2
        assert stats['completed_requests'] == 1
        assert stats['in_progress_requests'] == 1

    def test_request_info_to_dict(self):
        """Test RequestInfo to_dict method."""
        info = RequestInfo(
            request_id="test-123",
            paths=["/var/log/app.log"],
            patterns=["error.*"],
            max_results=100,
            started_at=datetime.now(),
            total_matches=50,
            hook_on_file_success=5,
            hook_on_file_failed=1,
        )

        data = info.model_dump(mode='json')
        assert data['request_id'] == 'test-123'
        assert data['total_matches'] == 50
        assert data['hooks']['on_file']['success'] == 5
        assert data['hooks']['on_file']['failed'] == 1


class TestHookCallbacks:
    """Tests for HookCallbacks dataclass."""

    def test_hook_callbacks_defaults(self):
        """Test HookCallbacks default values."""
        callbacks = HookCallbacks()
        assert callbacks.on_match_found is None
        assert callbacks.on_file_scanned is None
        assert callbacks.request_id == ""

    def test_hook_callbacks_with_values(self):
        """Test HookCallbacks with values."""

        def match_callback(payload):
            pass

        def file_callback(payload):
            pass

        callbacks = HookCallbacks(
            on_match_found=match_callback,
            on_file_scanned=file_callback,
            request_id="test-123",
        )
        assert callbacks.on_match_found is match_callback
        assert callbacks.on_file_scanned is file_callback
        assert callbacks.request_id == "test-123"


class TestGetHookEnvConfig:
    """Tests for get_hook_env_config function."""

    def test_returns_dict_with_expected_keys(self):
        """Test returns dict with expected keys."""
        config = get_hook_env_config()
        assert 'on_file_url' in config
        assert 'on_match_url' in config
        assert 'on_complete_url' in config
        assert 'custom_hooks_disabled' in config
