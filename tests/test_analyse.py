"""Tests for file analysis functionality"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from rx.analyse import (
    FileAnalyzer,
    analyse_path,
    human_readable_size,
)
from rx.cli.analyse import analyse_command
from rx.models import AnalyseResponse, FileAnalysisResult
from rx.web import app


@pytest.fixture
def client():
    """Create test client"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def temp_text_file():
    """Create a temporary test file with known content"""
    content = """Line 1: Short line
Line 2: This is a much longer line with more content
Line 3: Medium length line here

Line 5: After empty line
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_empty_file():
    """Create an empty temporary file"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_binary_file():
    """Create a temporary binary file"""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
        f.write(b'\x00\x01\x02\x03\xff\xfe\xfd')
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_directory(temp_text_file, temp_binary_file):
    """Create a temporary directory with mixed files"""
    temp_dir = tempfile.mkdtemp()

    # Create a text file in the directory
    text_file = os.path.join(temp_dir, 'test.txt')
    with open(text_file, 'w') as f:
        f.write("Test content\n")

    # Create a binary file in the directory
    binary_file = os.path.join(temp_dir, 'test.bin')
    with open(binary_file, 'wb') as f:
        f.write(b'\x00\x01\x02')

    yield temp_dir

    # Cleanup
    for file in os.listdir(temp_dir):
        os.unlink(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)


class TestHumanReadableSize:
    """Tests for human_readable_size function"""

    def test_bytes(self):
        """Test bytes formatting"""
        assert human_readable_size(0) == "0.00 B"
        assert human_readable_size(512) == "512.00 B"
        assert human_readable_size(1023) == "1023.00 B"

    def test_kilobytes(self):
        """Test kilobytes formatting"""
        assert human_readable_size(1024) == "1.00 KB"
        assert human_readable_size(2048) == "2.00 KB"
        assert human_readable_size(1536) == "1.50 KB"

    def test_megabytes(self):
        """Test megabytes formatting"""
        assert human_readable_size(1024 * 1024) == "1.00 MB"
        assert human_readable_size(1024 * 1024 * 5) == "5.00 MB"

    def test_gigabytes(self):
        """Test gigabytes formatting"""
        assert human_readable_size(1024 * 1024 * 1024) == "1.00 GB"

    def test_terabytes(self):
        """Test terabytes formatting"""
        assert human_readable_size(1024 * 1024 * 1024 * 1024) == "1.00 TB"


class TestFileAnalyzer:
    """Tests for FileAnalyzer class"""

    def test_analyze_text_file(self, temp_text_file):
        """Test analyzing a text file"""
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(temp_text_file, "f1")

        assert result.file_id == "f1"
        assert result.filepath == temp_text_file
        assert result.is_text is True
        assert result.size_bytes > 0
        assert result.line_count == 5  # 5 lines (4 content + 1 empty)
        assert result.empty_line_count == 1
        assert result.line_length_max == 52  # "Line 2: This is a much longer line with more content"
        assert result.line_length_avg is not None
        assert result.line_length_median is not None
        assert result.line_length_p95 is not None
        assert result.line_length_p99 is not None
        assert result.line_length_stddev is not None
        assert result.line_length_max_line_number is not None
        assert result.line_length_max_byte_offset is not None
        assert result.line_ending is not None

    def test_analyze_empty_file(self, temp_empty_file):
        """Test analyzing an empty file"""
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(temp_empty_file, "f1")

        assert result.file_id == "f1"
        assert result.is_text is True
        assert result.size_bytes == 0
        assert result.line_count == 0
        assert result.empty_line_count == 0
        # Empty files return 0 instead of None
        assert result.line_length_max == 0
        assert result.line_length_avg == 0.0
        assert result.line_length_median == 0.0
        assert result.line_length_p95 == 0.0
        assert result.line_length_p99 == 0.0
        assert result.line_length_stddev == 0.0

    def test_analyze_binary_file(self, temp_binary_file):
        """Test analyzing a binary file"""
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(temp_binary_file, "f1")

        assert result.file_id == "f1"
        assert result.is_text is False
        assert result.size_bytes > 0
        # Text metrics should be None for binary files
        assert result.line_count is None
        assert result.empty_line_count is None
        assert result.line_length_max is None

    def test_file_hook(self, temp_text_file):
        """Test registering and executing file hooks"""
        analyzer = FileAnalyzer()
        hook_called = []

        def my_hook(filepath, result):
            hook_called.append(filepath)
            result.custom_metrics['hook_executed'] = True

        analyzer.register_file_hook(my_hook)
        result = analyzer.analyze_file(temp_text_file, "f1")

        assert len(hook_called) == 1
        assert hook_called[0] == temp_text_file
        assert result.custom_metrics.get('hook_executed') is True

    def test_line_hook(self, temp_text_file):
        """Test registering and executing line hooks"""
        analyzer = FileAnalyzer()
        lines_processed = []

        def my_hook(line, line_num, result):
            lines_processed.append((line_num, len(line)))

        analyzer.register_line_hook(my_hook)
        result = analyzer.analyze_file(temp_text_file, "f1")

        assert len(lines_processed) == 5  # 5 lines in file
        assert lines_processed[0][0] == 1  # First line number

    def test_post_hook(self, temp_text_file):
        """Test registering and executing post hooks"""
        analyzer = FileAnalyzer()
        post_hook_called = []

        def my_hook(result):
            post_hook_called.append(True)
            result.custom_metrics['post_processed'] = result.line_count

        analyzer.register_post_hook(my_hook)
        result = analyzer.analyze_file(temp_text_file, "f1")

        assert len(post_hook_called) == 1
        assert result.custom_metrics.get('post_processed') == result.line_count

    def test_multiple_hooks(self, temp_text_file):
        """Test multiple hooks of different types"""
        analyzer = FileAnalyzer()
        execution_order = []

        def file_hook(filepath, result):
            execution_order.append('file')

        def line_hook(line, line_num, result):
            execution_order.append(f'line_{line_num}')

        def post_hook(result):
            execution_order.append('post')

        analyzer.register_file_hook(file_hook)
        analyzer.register_line_hook(line_hook)
        analyzer.register_post_hook(post_hook)

        result = analyzer.analyze_file(temp_text_file, "f1")

        # File hook should run first, then line hooks, then post hook
        assert execution_order[0] == 'file'
        assert execution_order[-1] == 'post'
        assert 'line_1' in execution_order

    def test_metadata_fields(self, temp_text_file):
        """Test that metadata fields are populated"""
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(temp_text_file, "f1")

        # These should be populated for all files
        assert result.created_at is not None
        assert result.modified_at is not None
        assert result.permissions is not None
        # Owner might be None on some systems, so just check it exists
        assert hasattr(result, 'owner')


class TestAnalysePath:
    """Tests for analyse_path function"""

    def test_analyse_single_file(self, temp_text_file):
        """Test analyzing a single file"""
        result = analyse_path([temp_text_file])

        assert 'path' in result
        assert 'time' in result
        assert 'files' in result
        assert 'results' in result
        assert 'scanned_files' in result
        assert 'skipped_files' in result

        assert len(result['files']) == 1
        assert len(result['results']) == 1
        assert len(result['scanned_files']) == 1
        assert len(result['skipped_files']) == 0

        # Check file ID format
        assert 'f1' in result['files']
        assert result['files']['f1'] == temp_text_file

    def test_analyse_directory(self, temp_directory):
        """Test analyzing a directory - only text files are analyzed"""
        result = analyse_path([temp_directory])

        # Only text files are analyzed when scanning directories
        assert len(result['files']) == 1  # only test.txt
        assert len(result['results']) == 1
        assert len(result['scanned_files']) == 1
        # Binary file should be in skipped_files
        assert len(result['skipped_files']) == 1

    def test_analyse_multiple_paths(self, temp_text_file, temp_empty_file):
        """Test analyzing multiple paths"""
        result = analyse_path([temp_text_file, temp_empty_file])

        assert len(result['files']) == 2
        assert len(result['results']) == 2
        assert 'f1' in result['files']
        assert 'f2' in result['files']

    def test_analyse_with_max_workers(self, temp_directory):
        """Test analyzing with custom max_workers"""
        result = analyse_path([temp_directory], max_workers=2)

        # Only text files are analyzed
        assert len(result['files']) == 1
        assert result['time'] > 0

    def test_analyse_nonexistent_path(self):
        """Test analyzing a nonexistent path"""
        result = analyse_path(['/nonexistent/path'])

        # Should return empty results, not crash
        assert len(result['files']) == 0
        assert len(result['scanned_files']) == 0

    def test_timing_information(self, temp_text_file):
        """Test that timing information is included"""
        result = analyse_path([temp_text_file])

        assert 'time' in result
        assert isinstance(result['time'], float)
        assert result['time'] >= 0


class TestAnalyseEndpoint:
    """Tests for /v1/analyse API endpoint"""

    def test_analyse_requires_path(self, client):
        """Test analyse endpoint requires path parameter"""
        response = client.get('/v1/analyse')
        assert response.status_code == 422  # Validation error

    def test_analyse_single_file(self, client, temp_text_file):
        """Test analyzing a single file via API"""
        response = client.get('/v1/analyse', params={'path': temp_text_file})
        assert response.status_code == 200

        data = response.json()
        assert 'path' in data
        assert 'time' in data
        assert 'files' in data
        assert 'results' in data
        assert 'scanned_files' in data
        assert 'skipped_files' in data

        assert len(data['files']) == 1
        assert len(data['results']) == 1

    def test_analyse_with_max_workers(self, client, temp_text_file):
        """Test analyzing with custom max_workers parameter"""
        response = client.get('/v1/analyse', params={'path': temp_text_file, 'max_workers': 5})
        assert response.status_code == 200

    def test_analyse_nonexistent_file(self, client):
        """Test analyzing nonexistent file returns 404"""
        response = client.get('/v1/analyse', params={'path': '/nonexistent/file.txt'})
        assert response.status_code == 404

    def test_analyse_directory(self, client, temp_directory):
        """Test analyzing a directory via API"""
        response = client.get('/v1/analyse', params={'path': temp_directory})
        assert response.status_code == 200

        data = response.json()
        assert len(data['files']) >= 1  # At least one file in directory

    def test_analyse_response_structure(self, client, temp_text_file):
        """Test that response matches AnalyseResponse model"""
        response = client.get('/v1/analyse', params={'path': temp_text_file})
        assert response.status_code == 200

        data = response.json()

        # Validate response structure
        result = data['results'][0]
        assert 'file' in result
        assert 'size_bytes' in result
        assert 'size_human' in result
        assert 'is_text' in result
        assert 'created_at' in result
        assert 'modified_at' in result
        assert 'permissions' in result

        # Text file should have text metrics
        if result['is_text']:
            assert 'line_count' in result
            assert 'empty_line_count' in result

    def test_analyse_max_workers_validation(self, client, temp_text_file):
        """Test max_workers parameter validation"""
        # Too low
        response = client.get('/v1/analyse', params={'path': temp_text_file, 'max_workers': 0})
        assert response.status_code == 422

        # Too high
        response = client.get('/v1/analyse', params={'path': temp_text_file, 'max_workers': 100})
        assert response.status_code == 422


class TestAnalyseCLI:
    """Tests for rx analyse CLI command"""

    def test_analyse_requires_path(self):
        """Test analyse command requires path argument"""
        runner = CliRunner()
        result = runner.invoke(analyse_command, [])
        assert result.exit_code != 0

    def test_analyse_single_file(self, temp_text_file):
        """Test analyzing a single file via CLI"""
        runner = CliRunner()
        result = runner.invoke(analyse_command, [temp_text_file])
        assert result.exit_code == 0
        assert 'Analysis Results' in result.output or temp_text_file in result.output

    def test_analyse_json_output(self, temp_text_file):
        """Test --json flag outputs valid JSON"""
        runner = CliRunner()
        result = runner.invoke(analyse_command, [temp_text_file, '--json'])
        assert result.exit_code == 0

        # Should be valid JSON
        data = json.loads(result.output)
        assert 'path' in data
        assert 'files' in data
        assert 'results' in data

    def test_analyse_no_color(self, temp_text_file):
        """Test --no-color flag"""
        runner = CliRunner()
        result = runner.invoke(analyse_command, [temp_text_file, '--no-color'])
        assert result.exit_code == 0
        # Output should not contain ANSI escape codes
        assert '\x1b[' not in result.output or result.output.strip() == ''

    def test_analyse_max_workers(self, temp_text_file):
        """Test --max-workers parameter"""
        runner = CliRunner()
        result = runner.invoke(analyse_command, [temp_text_file, '--max-workers', '5'])
        assert result.exit_code == 0

    def test_analyse_multiple_paths(self, temp_text_file, temp_empty_file):
        """Test analyzing multiple paths"""
        runner = CliRunner()
        result = runner.invoke(analyse_command, [temp_text_file, temp_empty_file])
        assert result.exit_code == 0

    def test_analyse_directory(self, temp_directory):
        """Test analyzing a directory - only text files are analyzed"""
        runner = CliRunner()
        result = runner.invoke(analyse_command, [temp_directory])
        # Exit code 0 for success, 2 for warning (skipped binary files)
        assert result.exit_code in [0, 2]

    def test_analyse_nonexistent_file(self):
        """Test analyzing nonexistent file"""
        runner = CliRunner()
        result = runner.invoke(analyse_command, ['/nonexistent/file.txt'])
        assert result.exit_code != 0

    def test_analyse_with_skipped_files(self, temp_directory):
        """Test exit code when files are skipped"""
        runner = CliRunner()
        result = runner.invoke(analyse_command, [temp_directory])
        # Exit code 2 indicates warning (skipped files)
        assert result.exit_code in [0, 2]


class TestAnalyseModels:
    """Tests for Pydantic models"""

    def test_file_analysis_result_model(self):
        """Test FileAnalysisResult model creation"""
        result = FileAnalysisResult(
            file="f1",
            size_bytes=1024,
            size_human="1.00 KB",
            is_text=True,
            created_at="2024-01-01T00:00:00",
            modified_at="2024-01-01T00:00:00",
            permissions="0644",
            owner="user",
            line_count=10,
            empty_line_count=2,
            line_length_max=80,
            line_length_avg=40.5,
            line_length_median=42.0,
            line_length_p95=75.0,
            line_length_p99=79.0,
            line_length_stddev=10.2,
            line_length_max_line_number=5,
            line_length_max_byte_offset=100,
            line_ending="LF",
            custom_metrics={},
        )

        assert result.file == "f1"
        assert result.size_bytes == 1024
        assert result.is_text is True
        assert result.line_length_max == 80
        assert result.line_length_p95 == 75.0
        assert result.line_length_p99 == 79.0
        assert result.line_length_max_line_number == 5
        assert result.line_ending == "LF"

    def test_analyse_response_model(self):
        """Test AnalyseResponse model creation"""
        response = AnalyseResponse(
            path="/tmp/test",
            time=0.123,
            files={"f1": "/tmp/test/file.txt"},
            results=[
                FileAnalysisResult(
                    file="f1",
                    size_bytes=100,
                    size_human="100.00 B",
                    is_text=True,
                    created_at="2024-01-01T00:00:00",
                    modified_at="2024-01-01T00:00:00",
                    permissions="0644",
                    owner="user",
                    line_count=5,
                    empty_line_count=0,
                    line_length_max=20,
                    line_length_avg=15.0,
                    line_length_median=16.0,
                    line_length_p95=19.0,
                    line_length_p99=20.0,
                    line_length_stddev=2.5,
                    line_length_max_line_number=2,
                    line_length_max_byte_offset=50,
                    line_ending="LF",
                )
            ],
            scanned_files=["/tmp/test/file.txt"],
            skipped_files=[],
        )

        assert response.path == "/tmp/test"
        assert len(response.results) == 1
        assert len(response.files) == 1

    def test_analyse_response_to_cli(self):
        """Test AnalyseResponse.to_cli() method"""
        response = AnalyseResponse(
            path="/tmp/test",
            time=0.123,
            files={"f1": "/tmp/test/file.txt"},
            results=[
                FileAnalysisResult(
                    file="f1",
                    size_bytes=100,
                    size_human="100.00 B",
                    is_text=True,
                    created_at="2024-01-01T00:00:00",
                    modified_at="2024-01-01T00:00:00",
                    permissions="0644",
                    owner="user",
                    line_count=5,
                    empty_line_count=0,
                    line_length_max=20,
                    line_length_avg=15.0,
                    line_length_median=16.0,
                    line_length_p95=19.0,
                    line_length_p99=20.0,
                    line_length_stddev=2.5,
                    line_length_max_line_number=2,
                    line_length_max_byte_offset=50,
                    line_ending="LF",
                )
            ],
            scanned_files=["/tmp/test/file.txt"],
            skipped_files=[],
        )

        # Test without colors
        output = response.to_cli(colorize=False)
        assert isinstance(output, str)
        assert len(output) > 0
        # Verify new fields are in output
        assert "p95=" in output
        assert "p99=" in output
        assert "Line ending: LF" in output
        assert "Longest line:" in output

        # Test with colors
        colored_output = response.to_cli(colorize=True)
        assert isinstance(colored_output, str)
        assert len(colored_output) > 0
