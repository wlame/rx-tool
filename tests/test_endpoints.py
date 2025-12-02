"""Tests for FastAPI endpoints"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from rx.web import app


@pytest.fixture
def client():
    """Create test client"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def temp_test_file():
    """Create a temporary test file with known content"""
    content = """Line 1: Hello world
Line 2: Python is awesome
Line 3: FastAPI rocks
Line 4: Testing is important
Line 5: Hello again
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestHealthEndpoint:
    """Tests for the health/root endpoint (now at /)"""

    def test_health_returns_ok(self, client):
        """Test root health endpoint returns ok status"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'

    def test_health_includes_ripgrep_status(self, client):
        """Test health endpoint includes ripgrep availability"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'ripgrep_available' in data
        assert isinstance(data['ripgrep_available'], bool)

    def test_health_includes_app_version(self, client):
        """Test health endpoint includes app version"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'app_version' in data
        assert isinstance(data['app_version'], str)
        assert len(data['app_version']) > 0

    def test_health_includes_python_version(self, client):
        """Test health endpoint includes Python version"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'python_version' in data
        assert isinstance(data['python_version'], str)
        assert len(data['python_version']) > 0

    def test_health_includes_os_info(self, client):
        """Test health endpoint includes OS information"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'os_info' in data
        assert isinstance(data['os_info'], dict)
        assert 'system' in data['os_info']
        assert 'release' in data['os_info']
        assert 'version' in data['os_info']
        assert 'machine' in data['os_info']

    def test_health_includes_system_resources(self, client):
        """Test health endpoint includes system resources"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'system_resources' in data
        assert isinstance(data['system_resources'], dict)
        assert 'cpu_cores' in data['system_resources']
        assert 'cpu_cores_physical' in data['system_resources']
        assert 'ram_total_gb' in data['system_resources']
        assert 'ram_available_gb' in data['system_resources']
        assert 'ram_percent_used' in data['system_resources']
        assert data['system_resources']['cpu_cores'] > 0
        assert data['system_resources']['ram_total_gb'] > 0

    def test_health_includes_python_packages(self, client):
        """Test health endpoint includes Python package versions"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'python_packages' in data
        assert isinstance(data['python_packages'], dict)
        # Check for key packages
        assert 'fastapi' in data['python_packages']
        assert 'pydantic' in data['python_packages']

    def test_health_includes_constants(self, client):
        """Test health endpoint includes application constants"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'constants' in data
        assert isinstance(data['constants'], dict)
        # Check for key constants
        assert 'LOG_LEVEL' in data['constants']
        assert 'DEBUG_MODE' in data['constants']
        assert 'LINE_SIZE_ASSUMPTION_KB' in data['constants']
        assert 'MAX_SUBPROCESSES' in data['constants']
        assert 'MIN_CHUNK_SIZE_MB' in data['constants']
        assert 'MAX_FILES' in data['constants']
        # Verify types
        assert isinstance(data['constants']['LOG_LEVEL'], str)
        assert isinstance(data['constants']['DEBUG_MODE'], bool)
        assert isinstance(data['constants']['LINE_SIZE_ASSUMPTION_KB'], int)
        assert isinstance(data['constants']['MAX_SUBPROCESSES'], int)
        assert isinstance(data['constants']['MIN_CHUNK_SIZE_MB'], int)
        assert isinstance(data['constants']['MAX_FILES'], int)

    def test_health_includes_environment(self, client):
        """Test health endpoint includes environment variables"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'environment' in data
        assert isinstance(data['environment'], dict)
        # Environment dict may be empty if no app-related env vars are set

    def test_health_includes_docs_url(self, client):
        """Test health endpoint includes documentation URL"""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'docs_url' in data
        assert isinstance(data['docs_url'], str)
        assert data['docs_url'].startswith('http')


class TestTraceEndpoint:
    """Tests for the trace endpoint"""

    def test_trace_requires_filename(self, client):
        """Test trace endpoint requires filename parameter"""
        response = client.get('/v1/trace?regex=test')
        assert response.status_code == 422  # Validation error

    def test_trace_requires_regex(self, client):
        """Test trace endpoint requires regex parameter"""
        response = client.get('/v1/trace?path=test.txt')
        assert response.status_code == 422  # Validation error

    def test_trace_finds_matches(self, client, temp_test_file):
        """Test trace endpoint finds matching byte offsets"""
        response = client.get('/v1/trace', params={'path': temp_test_file, 'regexp': 'Hello'})
        assert response.status_code == 200
        data = response.json()
        # Check ID-based structure
        assert 'patterns' in data
        assert 'files' in data
        assert 'matches' in data
        assert len(data['patterns']) == 1  # p1: Hello
        assert len(data['files']) == 1  # f1: temp_test_file
        assert len(data['matches']) == 2  # Two matches for "Hello"
        assert all('pattern' in m and 'file' in m and 'offset' in m for m in data['matches'])

    def test_trace_no_matches(self, client, temp_test_file):
        """Test trace endpoint returns empty list when no matches"""
        response = client.get('/v1/trace', params={'path': temp_test_file, 'regexp': 'NOTFOUND'})
        assert response.status_code == 200
        data = response.json()
        assert data['matches'] == []

    def test_trace_with_regex_pattern(self, client, temp_test_file):
        """Test trace endpoint with regex pattern"""
        response = client.get('/v1/trace', params={'path': temp_test_file, 'regexp': r'Line \d+:'})
        assert response.status_code == 200
        data = response.json()
        # All 5 lines match this pattern
        assert len(data['matches']) == 5
        assert all('pattern' in m and 'file' in m and 'offset' in m for m in data['matches'])

    def test_trace_nonexistent_file(self, client):
        """Test trace endpoint with nonexistent file"""
        response = client.get('/v1/trace', params={'path': '/nonexistent/file.txt', 'regexp': 'test'})
        assert response.status_code == 404  # File not found

    def test_trace_invalid_regex(self, client, temp_test_file):
        """Test trace endpoint with invalid regex"""
        response = client.get('/v1/trace', params={'path': temp_test_file, 'regexp': '[invalid'})
        assert response.status_code == 400  # Invalid regex pattern

    def test_trace_binary_file(self, client):
        """Test trace endpoint skips binary files"""
        # Create a temporary binary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
            f.write(b'\x00\x01\x02\x03\xff\xfe\xfd')  # Binary data with null bytes
            binary_file = f.name

        try:
            response = client.get('/v1/trace', params={'path': binary_file, 'regexp': 'test'})
            # Binary files are skipped, not rejected - returns 200 with empty matches
            assert response.status_code == 200
            data = response.json()
            assert len(data['matches']) == 0
            assert len(data['skipped_files']) == 1
        finally:
            if os.path.exists(binary_file):
                os.unlink(binary_file)


class TestSamplesEndpoint:
    """Tests for the /v1/samples endpoint"""

    def test_samples_with_offsets(self, client, temp_test_file):
        """Test samples endpoint with byte offsets"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'offsets': '0,20'})
        assert response.status_code == 200
        data = response.json()
        assert data['path'] == temp_test_file
        # offsets now maps offset -> line number
        assert isinstance(data['offsets'], dict)
        assert '0' in data['offsets']
        assert '20' in data['offsets']
        assert data['offsets']['0'] >= 1  # Line number is 1-based
        assert data['lines'] == {}
        assert 'samples' in data
        assert '0' in data['samples']
        assert '20' in data['samples']

    def test_samples_with_lines(self, client, temp_test_file):
        """Test samples endpoint with line numbers"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': '1,3'})
        assert response.status_code == 200
        data = response.json()
        assert data['path'] == temp_test_file
        assert data['offsets'] == {}
        # lines now maps line number -> byte offset
        assert isinstance(data['lines'], dict)
        assert '1' in data['lines']
        assert '3' in data['lines']
        assert data['lines']['1'] >= 0  # Offset is 0-based
        assert 'samples' in data
        assert '1' in data['samples']
        assert '3' in data['samples']
        # Check content
        assert any('Hello world' in line for line in data['samples']['1'])
        assert any('FastAPI rocks' in line for line in data['samples']['3'])

    def test_samples_lines_single(self, client, temp_test_file):
        """Test samples endpoint with single line number"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': '2'})
        assert response.status_code == 200
        data = response.json()
        # lines now is a dict
        assert isinstance(data['lines'], dict)
        assert '2' in data['lines']
        assert data['lines']['2'] >= 0
        assert '2' in data['samples']
        assert any('Python is awesome' in line for line in data['samples']['2'])

    def test_samples_mutual_exclusivity(self, client, temp_test_file):
        """Test that offsets and lines cannot be used together"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'offsets': '0', 'lines': '1'})
        assert response.status_code == 400
        assert "cannot use both" in response.json()['detail'].lower()

    def test_samples_requires_offsets_or_lines(self, client, temp_test_file):
        """Test that either offsets or lines must be provided"""
        response = client.get('/v1/samples', params={'path': temp_test_file})
        assert response.status_code == 400
        assert "must provide" in response.json()['detail'].lower()

    def test_samples_with_context(self, client, temp_test_file):
        """Test samples endpoint with context parameter"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': '3', 'context': 1})
        assert response.status_code == 200
        data = response.json()
        assert data['before_context'] == 1
        assert data['after_context'] == 1
        # Should have line 2, 3, 4 (1 before, target, 1 after)
        assert len(data['samples']['3']) == 3

    def test_samples_with_before_after_context(self, client, temp_test_file):
        """Test samples endpoint with separate before/after context"""
        response = client.get(
            '/v1/samples', params={'path': temp_test_file, 'lines': '3', 'before_context': 1, 'after_context': 2}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['before_context'] == 1
        assert data['after_context'] == 2

    def test_samples_invalid_offsets_format(self, client, temp_test_file):
        """Test samples endpoint with invalid offsets format"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'offsets': 'invalid,data'})
        assert response.status_code == 400
        assert 'invalid offsets' in response.json()['detail'].lower()

    def test_samples_invalid_lines_format(self, client, temp_test_file):
        """Test samples endpoint with invalid lines format"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': 'abc,xyz'})
        assert response.status_code == 400
        assert 'invalid lines' in response.json()['detail'].lower()

    def test_samples_nonexistent_file(self, client):
        """Test samples endpoint with nonexistent file"""
        response = client.get('/v1/samples', params={'path': '/nonexistent/file.txt', 'offsets': '0'})
        assert response.status_code == 404

    def test_samples_line_beyond_file(self, client, temp_test_file):
        """Test samples endpoint with line number beyond file"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': '999'})
        assert response.status_code == 200
        data = response.json()
        # Should return empty samples for non-existent line
        assert data['samples']['999'] == []

    def test_samples_json_structure_with_lines(self, client, temp_test_file):
        """Test complete JSON structure when using lines"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': '2'})
        assert response.status_code == 200
        data = response.json()
        assert 'path' in data
        assert 'offsets' in data
        assert 'lines' in data
        assert 'before_context' in data
        assert 'after_context' in data
        assert 'samples' in data
        # offsets and lines are now dicts
        assert isinstance(data['offsets'], dict)
        assert isinstance(data['lines'], dict)
        assert isinstance(data['samples'], dict)

    def test_samples_json_structure_with_offsets(self, client, temp_test_file):
        """Test complete JSON structure when using offsets"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'offsets': '0'})
        assert response.status_code == 200
        data = response.json()
        # offsets now maps offset -> line number
        assert isinstance(data['offsets'], dict)
        assert '0' in data['offsets']
        assert data['lines'] == {}

    def test_samples_binary_file(self, client):
        """Test samples endpoint with binary file"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
            f.write(b'\x00\x01\x02\x03\xff\xfe\xfd')
            binary_file = f.name

        try:
            response = client.get('/v1/samples', params={'path': binary_file, 'offsets': '0'})
            assert response.status_code == 400  # Binary file rejected
        finally:
            if os.path.exists(binary_file):
                os.unlink(binary_file)
