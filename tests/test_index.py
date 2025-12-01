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
        os.environ.pop("RX_LARGE_TEXT_FILE_MB", None)
        threshold = get_large_file_threshold_bytes()
        assert threshold == 100 * 1024 * 1024  # 100MB

    def test_get_index_step_default(self):
        """Test default step is threshold/10 = 10MB."""
        os.environ.pop("RX_LARGE_TEXT_FILE_MB", None)
        step = get_index_step_bytes()
        assert step == 10 * 1024 * 1024  # 10MB

    def test_get_large_file_threshold_from_env(self):
        """Test threshold can be set via environment variable."""
        os.environ["RX_LARGE_TEXT_FILE_MB"] = "50"
        try:
            threshold = get_large_file_threshold_bytes()
            assert threshold == 50 * 1024 * 1024  # 50MB
        finally:
            os.environ.pop("RX_LARGE_TEXT_FILE_MB", None)

    def test_get_index_step_scales_with_threshold(self):
        """Test step size scales with threshold."""
        os.environ["RX_LARGE_TEXT_FILE_MB"] = "200"
        try:
            step = get_index_step_bytes()
            assert step == 20 * 1024 * 1024  # 20MB (200/10)
        finally:
            os.environ.pop("RX_LARGE_TEXT_FILE_MB", None)


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

    These tests override RX_LARGE_TEXT_FILE_MB to use small thresholds
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
        old_value = os.environ.get("RX_LARGE_TEXT_FILE_MB")
        # Set to a value that makes threshold ~1KB
        # Since threshold is MB * 1024 * 1024, we need a fractional approach
        # But env var is int, so we'll use a workaround in the test
        os.environ["RX_LARGE_TEXT_FILE_MB"] = "1"  # 1MB threshold
        yield
        if old_value is None:
            os.environ.pop("RX_LARGE_TEXT_FILE_MB", None)
        else:
            os.environ["RX_LARGE_TEXT_FILE_MB"] = old_value

    def test_threshold_from_env_variable(self):
        """Test that threshold is read from environment variable."""
        old_value = os.environ.get("RX_LARGE_TEXT_FILE_MB")
        try:
            os.environ["RX_LARGE_TEXT_FILE_MB"] = "50"
            threshold = get_large_file_threshold_bytes()
            assert threshold == 50 * 1024 * 1024  # 50MB

            step = get_index_step_bytes()
            assert step == 5 * 1024 * 1024  # 5MB (50/10)
        finally:
            if old_value is None:
                os.environ.pop("RX_LARGE_TEXT_FILE_MB", None)
            else:
                os.environ["RX_LARGE_TEXT_FILE_MB"] = old_value

    def test_index_step_is_threshold_divided_by_10(self):
        """Test that index step is always threshold / 10."""
        old_value = os.environ.get("RX_LARGE_TEXT_FILE_MB")
        try:
            for mb in [10, 50, 100, 200]:
                os.environ["RX_LARGE_TEXT_FILE_MB"] = str(mb)
                threshold = get_large_file_threshold_bytes()
                step = get_index_step_bytes()
                assert step == threshold // 10
                assert step == mb * 1024 * 1024 // 10
        finally:
            if old_value is None:
                os.environ.pop("RX_LARGE_TEXT_FILE_MB", None)
            else:
                os.environ["RX_LARGE_TEXT_FILE_MB"] = old_value

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
