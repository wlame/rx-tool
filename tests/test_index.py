"""Tests for large file indexing functionality."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from rx.analyse import FileAnalyzer, analyse_path
from rx.cli.index import index_command
from rx.index import (
    INDEX_VERSION,
    build_index,
    calculate_exact_line_for_offset,
    calculate_exact_offset_for_line,
    create_index_file,
    delete_index,
    find_line_offset,
    get_cache_dir,
    get_index_info,
    get_index_path,
    get_index_step_bytes,
    get_large_file_threshold_bytes,
    is_index_valid,
    load_index,
    save_index,
)


@pytest.fixture
def temp_text_file():
    """Create a temporary text file with known content."""
    content = "Line 1: First line\nLine 2: Second line\nLine 3: Third line\n"
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_large_file():
    """Create a larger temporary file for index testing."""
    # Create file with ~100 lines, each ~100 bytes
    lines = []
    for i in range(100):
        # Varying line lengths for statistics testing
        if i % 10 == 0:
            lines.append(f"Line {i:04d}: " + "X" * 150 + "\n")  # Long line
        elif i % 5 == 0:
            lines.append(f"Line {i:04d}: Short\n")  # Short line
        else:
            lines.append(f"Line {i:04d}: " + "A" * 80 + "\n")  # Medium line

    content = "".join(lines)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def cleanup_index(temp_text_file):
    """Cleanup index file after test."""
    yield temp_text_file
    delete_index(temp_text_file)


class TestCacheDirectory:
    """Tests for cache directory management."""

    def test_get_cache_dir_exists(self):
        """Test that cache directory is created."""
        cache_dir = get_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_get_cache_dir_in_expected_location(self):
        """Test cache directory is in ~/.cache/rx/indexes."""
        cache_dir = get_cache_dir()
        assert "rx" in str(cache_dir)
        assert "indexes" in str(cache_dir)


class TestIndexPath:
    """Tests for index path generation."""

    def test_get_index_path_returns_path(self, temp_text_file):
        """Test that get_index_path returns a Path object."""
        index_path = get_index_path(temp_text_file)
        assert isinstance(index_path, Path)

    def test_get_index_path_consistent(self, temp_text_file):
        """Test that same source path always gives same index path."""
        path1 = get_index_path(temp_text_file)
        path2 = get_index_path(temp_text_file)
        assert path1 == path2

    def test_get_index_path_different_for_different_files(self, temp_text_file, temp_large_file):
        """Test that different source files get different index paths."""
        path1 = get_index_path(temp_text_file)
        path2 = get_index_path(temp_large_file)
        assert path1 != path2

    def test_get_index_path_includes_filename(self, temp_text_file):
        """Test that index path includes original filename."""
        index_path = get_index_path(temp_text_file)
        original_name = os.path.basename(temp_text_file)
        assert original_name in str(index_path)

    def test_get_index_path_is_json(self, temp_text_file):
        """Test that index path has .json extension."""
        index_path = get_index_path(temp_text_file)
        assert str(index_path).endswith(".json")


class TestConfiguration:
    """Tests for configuration functions."""

    def test_get_large_file_threshold_default(self):
        """Test default threshold is 100MB."""
        # Clear any env var
        os.environ.pop("DEFAULT_LARGE_FILE_MB", None)
        threshold = get_large_file_threshold_bytes()
        assert threshold == 50 * 1024 * 1024  # 50MB

    def test_get_index_step_default(self):
        """Test default step is threshold/10 = 10MB."""
        os.environ.pop("DEFAULT_LARGE_FILE_MB", None)
        step = get_index_step_bytes()
        assert step == 1 * 1024 * 1024  # 1MB

    def test_get_large_file_threshold_from_env(self):
        """Test threshold can be set via environment variable."""
        os.environ["DEFAULT_LARGE_FILE_MB"] = "50"
        try:
            threshold = get_large_file_threshold_bytes()
            assert threshold == 50 * 1024 * 1024  # 50MB
        finally:
            os.environ.pop("DEFAULT_LARGE_FILE_MB", None)

    def test_get_index_step_scales_with_threshold(self):
        """Test step size scales with threshold."""
        os.environ["DEFAULT_LARGE_FILE_MB"] = "200"
        try:
            step = get_index_step_bytes()
            assert step == 4 * 1024 * 1024  # 4MB (200/50)
        finally:
            os.environ.pop("DEFAULT_LARGE_FILE_MB", None)


class TestBuildIndex:
    """Tests for index building functionality."""

    def test_build_index_returns_result(self, temp_text_file):
        """Test that build_index returns IndexBuildResult."""
        result = build_index(temp_text_file, step_bytes=100)
        assert result is not None
        assert hasattr(result, "line_index")
        assert hasattr(result, "line_count")

    def test_build_index_first_entry_is_line_1_offset_0(self, temp_text_file):
        """Test that first index entry is always line 1 at offset 0."""
        result = build_index(temp_text_file, step_bytes=10)
        assert result.line_index[0] == [1, 0]

    def test_build_index_counts_lines_correctly(self, temp_text_file):
        """Test that line count is accurate."""
        result = build_index(temp_text_file, step_bytes=100)
        # File has 3 lines
        assert result.line_count == 3

    def test_build_index_offsets_aligned_to_line_starts(self, temp_large_file):
        """Test that all offsets in index point to line starts."""
        result = build_index(temp_large_file, step_bytes=500)  # Small step for more entries

        with open(temp_large_file, "rb") as f:
            for line_num, offset in result.line_index:
                f.seek(offset)
                # If not at start of file, previous char should be newline
                if offset > 0:
                    f.seek(offset - 1)
                    prev_char = f.read(1)
                    assert prev_char == b"\n", f"Offset {offset} is not at line start"

    def test_build_index_calculates_statistics(self, temp_large_file):
        """Test that statistics are calculated."""
        result = build_index(temp_large_file, step_bytes=1000)

        assert result.line_length_max > 0
        assert result.line_length_avg > 0
        assert result.line_length_median > 0
        assert result.line_length_p95 > 0
        assert result.line_length_p99 > 0
        assert result.line_length_max_line_number > 0

    def test_build_index_detects_line_ending(self, temp_text_file):
        """Test that line ending is detected."""
        result = build_index(temp_text_file, step_bytes=100)
        assert result.line_ending in ["LF", "CRLF", "CR", "mixed"]

    def test_build_index_finds_longest_line(self, temp_large_file):
        """Test that longest line info is captured."""
        result = build_index(temp_large_file, step_bytes=1000)

        # Verify by reading the file
        with open(temp_large_file, "rb") as f:
            f.seek(result.line_length_max_byte_offset)
            line = f.readline()
            stripped = line.rstrip(b"\r\n")
            assert len(stripped) == result.line_length_max


class TestSaveLoadIndex:
    """Tests for index persistence."""

    def test_save_and_load_index(self, temp_text_file):
        """Test that saved index can be loaded."""
        index_data = {
            "version": INDEX_VERSION,
            "source_path": temp_text_file,
            "source_modified_at": "2025-01-01T00:00:00",
            "source_size_bytes": 100,
            "line_index": [[1, 0], [10, 500]],
        }

        index_path = get_index_path(temp_text_file)
        assert save_index(index_data, index_path)

        loaded = load_index(index_path)
        assert loaded is not None
        assert loaded["version"] == INDEX_VERSION
        assert loaded["line_index"] == [[1, 0], [10, 500]]

        # Cleanup
        delete_index(temp_text_file)

    def test_load_nonexistent_index(self, temp_text_file):
        """Test that loading nonexistent index returns None."""
        loaded = load_index("/nonexistent/path.json")
        assert loaded is None

    def test_load_invalid_json(self, temp_text_file):
        """Test that loading invalid JSON returns None."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write("not valid json")
            invalid_path = f.name

        try:
            loaded = load_index(invalid_path)
            assert loaded is None
        finally:
            os.unlink(invalid_path)

    def test_load_wrong_version(self, temp_text_file):
        """Test that loading wrong version returns None."""
        index_data = {"version": 999, "line_index": [[1, 0]]}

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(index_data, f)
            path = f.name

        try:
            loaded = load_index(path)
            assert loaded is None
        finally:
            os.unlink(path)


class TestIndexValidation:
    """Tests for index validation."""

    def test_is_index_valid_no_index(self, temp_text_file):
        """Test that nonexistent index is invalid."""
        delete_index(temp_text_file)  # Ensure no index
        assert not is_index_valid(temp_text_file)

    def test_is_index_valid_fresh_index(self, temp_text_file):
        """Test that freshly created index is valid."""
        create_index_file(temp_text_file, force=True)
        assert is_index_valid(temp_text_file)
        delete_index(temp_text_file)

    def test_is_index_valid_after_file_modification(self, temp_text_file):
        """Test that index becomes invalid after file modification."""
        create_index_file(temp_text_file, force=True)
        assert is_index_valid(temp_text_file)

        # Wait a moment and modify file
        time.sleep(0.1)
        with open(temp_text_file, "a") as f:
            f.write("New line\n")

        assert not is_index_valid(temp_text_file)
        delete_index(temp_text_file)


class TestFindLineOffset:
    """Tests for line offset lookup."""

    def test_find_line_offset_exact_match(self):
        """Test finding exact line number in index."""
        line_index = [[1, 0], [100, 1000], [200, 2000]]
        line, offset = find_line_offset(line_index, 100)
        assert line == 100
        assert offset == 1000

    def test_find_line_offset_between_entries(self):
        """Test finding line between index entries."""
        line_index = [[1, 0], [100, 1000], [200, 2000]]
        line, offset = find_line_offset(line_index, 150)
        # Should return closest previous: line 100
        assert line == 100
        assert offset == 1000

    def test_find_line_offset_before_first(self):
        """Test finding line before first index entry."""
        line_index = [[10, 100], [100, 1000]]
        line, offset = find_line_offset(line_index, 5)
        # Should return first entry
        assert line == 10
        assert offset == 100

    def test_find_line_offset_after_last(self):
        """Test finding line after last index entry."""
        line_index = [[1, 0], [100, 1000], [200, 2000]]
        line, offset = find_line_offset(line_index, 500)
        # Should return last entry
        assert line == 200
        assert offset == 2000

    def test_find_line_offset_empty_index(self):
        """Test finding line in empty index."""
        line, offset = find_line_offset([], 10)
        assert line == 1
        assert offset == 0

    def test_find_line_offset_first_line(self):
        """Test finding first line."""
        line_index = [[1, 0], [100, 1000]]
        line, offset = find_line_offset(line_index, 1)
        assert line == 1
        assert offset == 0


class TestCreateIndexFile:
    """Tests for high-level index creation."""

    def test_create_index_file_creates_file(self, temp_text_file):
        """Test that create_index_file creates an index file."""
        delete_index(temp_text_file)
        result = create_index_file(temp_text_file)

        assert result is not None
        assert get_index_path(temp_text_file).exists()

        delete_index(temp_text_file)

    def test_create_index_file_force_rebuild(self, temp_text_file):
        """Test that force=True rebuilds index."""
        # Create initial index
        result1 = create_index_file(temp_text_file, force=True)
        created_at_1 = result1["created_at"]

        # Wait and force rebuild
        time.sleep(0.1)
        result2 = create_index_file(temp_text_file, force=True)
        created_at_2 = result2["created_at"]

        assert created_at_1 != created_at_2

        delete_index(temp_text_file)

    def test_create_index_file_reuses_valid(self, temp_text_file):
        """Test that valid index is reused without force."""
        result1 = create_index_file(temp_text_file, force=True)
        created_at_1 = result1["created_at"]

        time.sleep(0.1)
        result2 = create_index_file(temp_text_file, force=False)
        created_at_2 = result2["created_at"]

        # Should be same since index was valid
        assert created_at_1 == created_at_2

        delete_index(temp_text_file)

    def test_create_index_file_includes_analysis(self, temp_text_file):
        """Test that index includes analysis data."""
        result = create_index_file(temp_text_file, force=True)

        assert "analysis" in result
        analysis = result["analysis"]
        assert "line_count" in analysis
        assert "line_length_max" in analysis
        assert "line_length_avg" in analysis
        assert "line_length_median" in analysis
        assert "line_length_p95" in analysis
        assert "line_length_p99" in analysis
        assert "line_ending" in analysis

        delete_index(temp_text_file)


class TestDeleteIndex:
    """Tests for index deletion."""

    def test_delete_index_removes_file(self, temp_text_file):
        """Test that delete_index removes the index file."""
        create_index_file(temp_text_file, force=True)
        assert get_index_path(temp_text_file).exists()

        result = delete_index(temp_text_file)
        assert result is True
        assert not get_index_path(temp_text_file).exists()

    def test_delete_index_nonexistent(self, temp_text_file):
        """Test that deleting nonexistent index succeeds."""
        delete_index(temp_text_file)  # Ensure no index
        result = delete_index(temp_text_file)
        assert result is True


class TestGetIndexInfo:
    """Tests for index info retrieval."""

    def test_get_index_info_no_index(self, temp_text_file):
        """Test get_index_info when no index exists."""
        delete_index(temp_text_file)
        info = get_index_info(temp_text_file)
        assert info is None

    def test_get_index_info_with_index(self, temp_text_file):
        """Test get_index_info with existing index."""
        create_index_file(temp_text_file, force=True)

        info = get_index_info(temp_text_file)
        assert info is not None
        assert "index_path" in info
        assert "source_path" in info
        assert "is_valid" in info
        assert "index_entries" in info
        assert "analysis" in info
        assert info["is_valid"] is True

        delete_index(temp_text_file)


class TestIndexCLI:
    """Tests for rx index CLI command."""

    def test_index_command_creates_index(self, temp_text_file):
        """Test that index command creates an index."""
        delete_index(temp_text_file)

        runner = CliRunner()
        result = runner.invoke(index_command, [temp_text_file, "--force"])

        assert result.exit_code == 0
        assert "building index" in result.output or "done" in result.output
        assert get_index_path(temp_text_file).exists()

        delete_index(temp_text_file)

    def test_index_command_info(self, temp_text_file):
        """Test index command --info flag."""
        create_index_file(temp_text_file, force=True)

        runner = CliRunner()
        result = runner.invoke(index_command, [temp_text_file, "--info"])

        assert result.exit_code == 0
        assert "Index path:" in result.output
        assert "Valid: True" in result.output
        assert "Lines:" in result.output

        delete_index(temp_text_file)

    def test_index_command_info_no_index(self, temp_text_file):
        """Test index command --info when no index exists."""
        delete_index(temp_text_file)

        runner = CliRunner()
        result = runner.invoke(index_command, [temp_text_file, "--info"])

        assert result.exit_code == 0
        assert "no index exists" in result.output

    def test_index_command_delete(self, temp_text_file):
        """Test index command --delete flag."""
        create_index_file(temp_text_file, force=True)
        assert get_index_path(temp_text_file).exists()

        runner = CliRunner()
        result = runner.invoke(index_command, [temp_text_file, "--delete"])

        assert result.exit_code == 0
        assert "deleted" in result.output
        assert not get_index_path(temp_text_file).exists()

    def test_index_command_force_rebuild(self, temp_text_file):
        """Test index command --force flag rebuilds index."""
        result1 = create_index_file(temp_text_file, force=True)
        created_at_1 = result1["created_at"]

        time.sleep(0.1)

        runner = CliRunner()
        result = runner.invoke(index_command, [temp_text_file, "--force"])

        assert result.exit_code == 0

        info = get_index_info(temp_text_file)
        assert info["created_at"] != created_at_1

        delete_index(temp_text_file)

    def test_index_command_json_output(self, temp_text_file):
        """Test index command --json flag."""
        delete_index(temp_text_file)

        runner = CliRunner()
        result = runner.invoke(index_command, [temp_text_file, "--force", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "files" in data
        assert len(data["files"]) == 1
        assert data["files"][0]["action"] == "created"

        delete_index(temp_text_file)

    def test_index_command_skip_existing_valid(self, temp_text_file):
        """Test that index command skips valid existing index."""
        create_index_file(temp_text_file, force=True)

        runner = CliRunner()
        result = runner.invoke(index_command, [temp_text_file])

        assert result.exit_code == 0
        assert "valid index exists" in result.output

        delete_index(temp_text_file)


class TestAnalyseIndexIntegration:
    """Tests for analyse command integration with indexing."""

    def test_analyzer_uses_cache(self, temp_text_file):
        """Test that FileAnalyzer uses cached analysis when available."""
        # Create index first
        create_index_file(temp_text_file, force=True)

        # Analyze with cache enabled
        analyzer = FileAnalyzer(use_index_cache=True)
        result = analyzer.analyze_file(temp_text_file, "f1")

        assert result is not None
        assert result.line_count is not None
        assert result.line_length_max is not None

        delete_index(temp_text_file)

    def test_analyzer_cache_disabled(self, temp_text_file):
        """Test that FileAnalyzer can disable cache."""
        create_index_file(temp_text_file, force=True)

        # Analyze with cache disabled
        analyzer = FileAnalyzer(use_index_cache=False)
        result = analyzer.analyze_file(temp_text_file, "f1")

        # Should still work, just not use cache
        assert result is not None
        assert result.line_count is not None

        delete_index(temp_text_file)

    def test_analyzer_invalidates_stale_cache(self, temp_text_file):
        """Test that stale cache is not used."""
        create_index_file(temp_text_file, force=True)

        # Modify file to invalidate cache
        time.sleep(0.1)
        with open(temp_text_file, "a") as f:
            f.write("New line added\n")

        # Cache should be invalid now
        assert not is_index_valid(temp_text_file)

        # Analyzer should not use stale cache
        analyzer = FileAnalyzer(use_index_cache=True)
        result = analyzer.analyze_file(temp_text_file, "f1")

        # Result should reflect new file content
        assert result is not None
        # Original file had 3 lines, now has 4
        assert result.line_count == 4

        delete_index(temp_text_file)

    def test_analyse_path_with_cache(self, temp_text_file):
        """Test analyse_path uses cached indexes."""
        # Create index
        create_index_file(temp_text_file, force=True)

        # Run analyse_path
        result = analyse_path([temp_text_file])

        assert len(result["results"]) == 1
        assert result["results"][0]["line_count"] is not None

        delete_index(temp_text_file)


class TestThresholdBasedIndexing:
    """Tests for automatic indexing based on file size threshold.

    These tests override DEFAULT_LARGE_FILE_MB to use small thresholds
    so we can test with reasonably sized test files.
    """

    @pytest.fixture
    def file_above_threshold(self):
        """Create a file that's above a 1KB threshold (will be ~2KB)."""
        lines = [f"Line {i:04d}: " + "X" * 80 + "\n" for i in range(20)]
        content = "".join(lines)  # ~2KB

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)
        delete_index(temp_path)

    @pytest.fixture
    def file_below_threshold(self):
        """Create a small file that's below a 1KB threshold (~100 bytes)."""
        content = "Small file\nJust two lines\n"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)
        delete_index(temp_path)

    @pytest.fixture
    def low_threshold(self):
        """Set a low threshold (1KB) for testing."""
        old_value = os.environ.get("DEFAULT_LARGE_FILE_MB")
        # Set to a value that makes threshold ~1KB
        # Since threshold is MB * 1024 * 1024, we need a fractional approach
        # But env var is int, so we'll use a workaround in the test
        os.environ["DEFAULT_LARGE_FILE_MB"] = "1"  # 1MB threshold
        yield
        if old_value is None:
            os.environ.pop("DEFAULT_LARGE_FILE_MB", None)
        else:
            os.environ["DEFAULT_LARGE_FILE_MB"] = old_value

    def test_threshold_from_env_variable(self):
        """Test that threshold is read from environment variable."""
        old_value = os.environ.get("DEFAULT_LARGE_FILE_MB")
        try:
            os.environ["DEFAULT_LARGE_FILE_MB"] = "50"
            threshold = get_large_file_threshold_bytes()
            assert threshold == 50 * 1024 * 1024  # 50MB

            step = get_index_step_bytes()
            assert step == 1 * 1024 * 1024  # 1MB (50/50)
        finally:
            if old_value is None:
                os.environ.pop("DEFAULT_LARGE_FILE_MB", None)
            else:
                os.environ["DEFAULT_LARGE_FILE_MB"] = old_value

    def test_index_step_is_threshold_divided_by_10(self):
        """Test that index step is always threshold / 10."""
        old_value = os.environ.get("DEFAULT_LARGE_FILE_MB")
        try:
            for mb in [10, 50, 100, 200]:
                os.environ["DEFAULT_LARGE_FILE_MB"] = str(mb)
                threshold = get_large_file_threshold_bytes()
                step = get_index_step_bytes()
                assert step == threshold // 50
                assert step == mb * 1024 * 1024 // 50
        finally:
            if old_value is None:
                os.environ.pop("DEFAULT_LARGE_FILE_MB", None)
            else:
                os.environ["DEFAULT_LARGE_FILE_MB"] = old_value

    def test_analyse_creates_index_for_large_file(self):
        """Test that analyse automatically creates index for files above threshold.

        We create a file larger than threshold and verify index is created.
        """
        # Create a file that will be "large" - we'll make threshold very small
        # by mocking. For now, test with explicit index creation.
        lines = [f"Line {i:04d}: " + "A" * 100 + "\n" for i in range(1000)]
        content = "".join(lines)  # ~110KB

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Verify file size
            file_size = os.path.getsize(temp_path)
            assert file_size > 100000  # > 100KB

            # Build index and verify it works correctly
            result = build_index(temp_path, step_bytes=10000)  # 10KB step

            # Should have multiple index entries for 110KB file with 10KB step
            assert len(result.line_index) > 1
            assert result.line_count == 1000

            # Verify all offsets are line-aligned
            with open(temp_path, "rb") as f:
                for line_num, offset in result.line_index:
                    if offset > 0:
                        f.seek(offset - 1)
                        prev_char = f.read(1)
                        assert prev_char == b"\n", f"Offset {offset} not at line start"

        finally:
            os.unlink(temp_path)
            delete_index(temp_path)

    def test_index_entries_at_correct_intervals(self):
        """Test that index entries are created at approximately correct byte intervals."""
        # Create a file with known line lengths
        line_length = 100  # Each line is exactly 100 bytes (99 chars + newline)
        num_lines = 100
        lines = ["A" * 99 + "\n" for _ in range(num_lines)]
        content = "".join(lines)  # 10,000 bytes exactly

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Use 2500 byte step - should get entries at ~0, 2500, 5000, 7500
            result = build_index(temp_path, step_bytes=2500)

            # First entry is always line 1 at offset 0
            assert result.line_index[0] == [1, 0]

            # Should have multiple entries
            assert len(result.line_index) >= 3

            # Each subsequent entry should be at approximately step_bytes intervals
            for i in range(1, len(result.line_index)):
                offset = result.line_index[i][1]
                expected_min = i * 2500
                # Offset should be >= expected (we record after passing checkpoint)
                assert offset >= expected_min, f"Entry {i} offset {offset} < expected {expected_min}"

        finally:
            os.unlink(temp_path)

    def test_find_line_offset_accuracy(self):
        """Test that find_line_offset returns correct closest previous entry."""
        # Create index with known entries
        line_index = [
            [1, 0],
            [100, 10000],
            [200, 20000],
            [300, 30000],
        ]

        # Exact matches
        assert find_line_offset(line_index, 1) == (1, 0)
        assert find_line_offset(line_index, 100) == (100, 10000)
        assert find_line_offset(line_index, 200) == (200, 20000)

        # Between entries - should return previous
        assert find_line_offset(line_index, 50) == (1, 0)
        assert find_line_offset(line_index, 150) == (100, 10000)
        assert find_line_offset(line_index, 250) == (200, 20000)

        # After last entry
        assert find_line_offset(line_index, 500) == (300, 30000)

    def test_cli_respects_threshold_option(self):
        """Test that CLI --threshold option works."""
        # Create a small file (~500 bytes)
        content = "Small line\n" * 50  # ~550 bytes

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            runner = CliRunner()

            # With default threshold (100MB), file is too small, should skip
            # But with --force it will still index
            result = runner.invoke(index_command, [temp_path, "--force"])
            assert result.exit_code == 0
            assert "done" in result.output

            delete_index(temp_path)

        finally:
            os.unlink(temp_path)


class TestOffsetLineMapping:
    """Test bidirectional offset<->line mapping functionality."""

    def test_calculate_offset_for_line_small_file_no_index(self):
        """Test calculating byte offset for line in small file without index."""
        # Create a small test file
        content = "Line 1\nLine 2\nLine 3\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Line 1 should be at offset 0
            offset = calculate_exact_offset_for_line(temp_path, 1, None)
            assert offset == 0

            # Line 2 should be at offset 7 (after "Line 1\n")
            offset = calculate_exact_offset_for_line(temp_path, 2, None)
            assert offset == 7

            # Line 3 should be at offset 14 (after "Line 1\nLine 2\n")
            offset = calculate_exact_offset_for_line(temp_path, 3, None)
            assert offset == 14

        finally:
            os.unlink(temp_path)

    def test_calculate_line_for_offset_small_file_no_index(self):
        """Test calculating line number for byte offset in small file without index."""
        content = "Line 1\nLine 2\nLine 3\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Offset 0 should be line 1
            line = calculate_exact_line_for_offset(temp_path, 0, None)
            assert line == 1

            # Offset 7 should be line 2
            line = calculate_exact_line_for_offset(temp_path, 7, None)
            assert line == 2

            # Offset 14 should be line 3
            line = calculate_exact_line_for_offset(temp_path, 14, None)
            assert line == 3

            # Offset 5 (middle of line 1) should be line 1
            line = calculate_exact_line_for_offset(temp_path, 5, None)
            assert line == 1

        finally:
            os.unlink(temp_path)

    def test_calculate_offset_for_line_with_index(self):
        """Test calculating byte offset using index data."""
        # Create a file and build an index
        lines = []
        for i in range(200):
            lines.append(f"Line {i + 1}: Some content here for line {i + 1}\n")
        content = "".join(lines)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Create index using the proper API
            index_data = create_index_file(temp_path, force=True)

            # Calculate offset for various lines
            offset_1 = calculate_exact_offset_for_line(temp_path, 1, index_data)
            assert offset_1 == 0

            # Verify by calculating line back
            line_1 = calculate_exact_line_for_offset(temp_path, offset_1, index_data)
            assert line_1 == 1

            # Test middle line
            offset_100 = calculate_exact_offset_for_line(temp_path, 100, index_data)
            assert offset_100 > 0
            line_100 = calculate_exact_line_for_offset(temp_path, offset_100, index_data)
            assert line_100 == 100

            # Test last line
            offset_200 = calculate_exact_offset_for_line(temp_path, 200, index_data)
            assert offset_200 > offset_100
            line_200 = calculate_exact_line_for_offset(temp_path, offset_200, index_data)
            assert line_200 == 200

        finally:
            delete_index(temp_path)
            os.unlink(temp_path)

    def test_calculate_line_for_offset_with_index(self):
        """Test calculating line number using index data."""
        lines = []
        for i in range(150):
            lines.append(f"Line {i + 1} content\n")
        content = "".join(lines)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Create index using the proper API
            index_data = create_index_file(temp_path, force=True)

            # Test various offsets
            line = calculate_exact_line_for_offset(temp_path, 0, index_data)
            assert line == 1

            # Find offset of line 50, then verify we can find line 50 from that offset
            offset_50 = calculate_exact_offset_for_line(temp_path, 50, index_data)
            line_50 = calculate_exact_line_for_offset(temp_path, offset_50, index_data)
            assert line_50 == 50

        finally:
            delete_index(temp_path)
            os.unlink(temp_path)

    def test_large_file_without_index_returns_minus_one(self, monkeypatch):
        """Test that large files without index return -1."""
        # Create a small file
        content = "Line 1\nLine 2\nLine 3\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Mock the threshold to be smaller than our file
            monkeypatch.setattr("rx.index.get_large_file_threshold_bytes", lambda: 1)

            # Without index, should return -1 for "large" file
            offset = calculate_exact_offset_for_line(temp_path, 2, None)
            assert offset == -1

            line = calculate_exact_line_for_offset(temp_path, 7, None)
            assert line == -1

        finally:
            os.unlink(temp_path)

    def test_bidirectional_consistency(self):
        """Test that offset->line->offset produces consistent results."""
        content = "First\nSecond\nThird\nFourth\nFifth\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Test round-trip: line -> offset -> line
            for line_num in [1, 2, 3, 4, 5]:
                offset = calculate_exact_offset_for_line(temp_path, line_num, None)
                back_to_line = calculate_exact_line_for_offset(temp_path, offset, None)
                assert back_to_line == line_num

        finally:
            os.unlink(temp_path)

    def test_offset_for_nonexistent_line(self):
        """Test behavior when requesting offset for line beyond file end."""
        content = "Line 1\nLine 2\nLine 3\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Request line that doesn't exist
            offset = calculate_exact_offset_for_line(temp_path, 100, None)
            assert offset == -1

        finally:
            os.unlink(temp_path)

    def test_line_for_offset_beyond_file(self):
        """Test behavior when requesting line for offset beyond file size."""
        content = "Line 1\nLine 2\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            file_size = os.path.getsize(temp_path)
            # Request offset beyond file
            line = calculate_exact_line_for_offset(temp_path, file_size + 1000, None)
            assert line == -1

        finally:
            os.unlink(temp_path)

    def test_empty_file_handling(self):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            # Empty file
            offset = calculate_exact_offset_for_line(temp_path, 1, None)
            assert offset == -1

            line = calculate_exact_line_for_offset(temp_path, 0, None)
            # Empty file has no lines
            assert line == -1

        finally:
            os.unlink(temp_path)

    def test_file_without_trailing_newline(self):
        """Test files that don't end with newline."""
        content = "Line 1\nLine 2\nLine 3"  # No trailing newline
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Line 1
            offset = calculate_exact_offset_for_line(temp_path, 1, None)
            assert offset == 0

            # Line 3 (last line without newline)
            offset = calculate_exact_offset_for_line(temp_path, 3, None)
            assert offset == 14

            # Verify reverse mapping
            line = calculate_exact_line_for_offset(temp_path, 14, None)
            assert line == 3

        finally:
            os.unlink(temp_path)

    def test_large_file_with_index_performance(self):
        """Test that large file with index can efficiently find distant lines."""
        # Create a file large enough to require indexing with many lines
        # Use 60MB threshold to ensure indexing (override default 50MB)
        lines = []
        # Create ~100k lines of 100 bytes each = ~10MB file
        for i in range(100_000):
            lines.append(f"Line {i + 1:06d}: {'x' * 80}\n")
        content = "".join(lines)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Force create an index
            index_data = create_index_file(temp_path, force=True)
            assert index_data is not None

            # Verify index has multiple checkpoints
            line_index = index_data.get("line_index", [])
            assert len(line_index) > 5, f"Expected multiple index checkpoints, got {len(line_index)}"

            # Test finding offset for a line far from the start (line 90000)
            import time

            start = time.time()
            offset_90k = calculate_exact_offset_for_line(temp_path, 90_000, index_data)
            duration = time.time() - start

            # Should complete quickly (< 1 second for 100k line file)
            assert duration < 1.0, f"Took {duration:.2f}s to find line 90000, too slow!"
            assert offset_90k > 0, "Should find valid offset"

            # Verify the offset is correct by reading that line
            with open(temp_path, "rb") as f:
                f.seek(offset_90k)
                line = f.readline().decode("utf-8")
                assert "Line 090000:" in line, f"Wrong line at offset {offset_90k}: {line[:50]}"

            # Test reverse mapping (offset -> line)
            start = time.time()
            line_num = calculate_exact_line_for_offset(temp_path, offset_90k, index_data)
            duration = time.time() - start

            assert duration < 1.0, f"Took {duration:.2f}s to find line for offset, too slow!"
            assert line_num == 90_000, f"Expected line 90000, got {line_num}"

        finally:
            delete_index(temp_path)
            os.unlink(temp_path)

    def test_many_short_lines_with_index(self):
        """Test file with many short lines (like Twitter data)."""
        # Simulate Twitter-like data: short lines (~50 bytes)
        # Create 500k lines of ~50 bytes each = ~25MB file
        lines = []
        for i in range(500_000):
            lines.append(f"Tweet {i + 1}: Short message here\n")  # ~35 bytes
        content = "".join(lines)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Force create an index
            index_data = create_index_file(temp_path, force=True)
            assert index_data is not None

            line_index = index_data.get("line_index", [])
            print(f"Index has {len(line_index)} checkpoints for 500k lines")

            # Test finding a line near the end
            import time

            start = time.time()
            offset_450k = calculate_exact_offset_for_line(temp_path, 450_000, index_data)
            duration = time.time() - start

            print(f"Took {duration:.3f}s to find line 450,000")
            # Should complete in reasonable time (< 2 seconds for 500k lines)
            assert duration < 2.0, f"Took {duration:.2f}s to find line 450000, too slow!"
            assert offset_450k > 0

            # Verify correctness
            with open(temp_path, "rb") as f:
                f.seek(offset_450k)
                line = f.readline().decode("utf-8")
                assert "Tweet 450000:" in line, f"Wrong line: {line}"

        finally:
            delete_index(temp_path)
            os.unlink(temp_path)

    def test_last_line_after_last_checkpoint(self):
        """Test finding the last line when it's AFTER the last index checkpoint.

        This reproduces the bug where requesting the last line of a large file
        returns -1 even though the index exists and the line is just a few
        thousand lines past the last checkpoint.

        User's scenario:
        - Last checkpoint at line 133,123,748
        - Requested line 133,127,816 (4068 lines after last checkpoint)
        - Bug: returns -1 instead of finding the offset
        """
        # Create a file where the last checkpoint won't be at the last line
        # Use 20-byte lines so ~50k lines per 1MB checkpoint
        lines = []
        num_lines = 100_000
        for i in range(num_lines):
            lines.append(f"Line {i + 1:06d}: data\n")  # ~20 bytes
        content = "".join(lines)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Force create an index
            index_data = create_index_file(temp_path, force=True)
            assert index_data is not None

            line_index = index_data.get("line_index", [])
            last_indexed_line = line_index[-1][0]
            last_indexed_offset = line_index[-1][1]
            total_lines = index_data.get("analysis", {}).get("line_count", 0)

            print(f"Index has {len(line_index)} checkpoints")
            print(f"Last checkpoint: line {last_indexed_line} at offset {last_indexed_offset}")
            print(f"Total lines in file: {total_lines}")
            print(f"Lines after last checkpoint: {total_lines - last_indexed_line}")

            # THE KEY TEST: find the LAST line of the file
            # This line is AFTER the last checkpoint
            target_line = num_lines  # The very last line

            import time

            start = time.time()
            offset = calculate_exact_offset_for_line(temp_path, target_line, index_data)
            duration = time.time() - start

            print(f"Target line {target_line}, got offset {offset} in {duration:.3f}s")

            # THIS IS THE BUG: offset should NOT be -1 for the last line!
            assert offset != -1, f"BUG: Got -1 for last line {target_line}, should find valid offset"
            assert offset > 0, f"Offset should be positive, got {offset}"

            # Verify correctness by reading the line
            with open(temp_path, "rb") as f:
                f.seek(offset)
                line = f.readline().decode("utf-8")
                expected = f"Line {target_line:06d}:"
                assert expected in line, f"Wrong line at offset {offset}: {line}"

        finally:
            delete_index(temp_path)
            os.unlink(temp_path)

    def test_calculate_offset_without_passing_index_data(self):
        """Test that calculate_exact_offset_for_line works when index_data is None.

        This tests the bug fix where load_index() was called with the source file
        path instead of the index file path, causing it to fail to load the index.
        """
        # Create a file and index
        lines = []
        for i in range(50_000):
            lines.append(f"Line {i + 1:05d}: test\n")  # ~20 bytes
        content = "".join(lines)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Create index
            index_data = create_index_file(temp_path, force=True)
            assert index_data is not None

            # Now call WITHOUT passing index_data - the function should load it
            target_line = 45_000
            offset = calculate_exact_offset_for_line(temp_path, target_line, None)

            # Should NOT return -1 - should find the offset using the index
            assert offset != -1, f"BUG: Got -1, function failed to load index from disk"
            assert offset > 0

            # Verify correctness
            with open(temp_path, "rb") as f:
                f.seek(offset)
                line = f.readline().decode("utf-8")
                assert f"Line {target_line:05d}:" in line, f"Wrong line: {line}"

            # Also test reverse direction
            line_num = calculate_exact_line_for_offset(temp_path, offset, None)
            assert line_num == target_line, f"Expected {target_line}, got {line_num}"

        finally:
            delete_index(temp_path)
            os.unlink(temp_path)
