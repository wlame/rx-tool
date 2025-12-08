"""Tests for compressed file index management."""

import gzip
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rx.compressed_index import (
    INDEX_VERSION,
    build_compressed_index,
    clear_compressed_indexes,
    delete_compressed_index,
    find_nearest_checkpoint,
    get_compressed_index_dir,
    get_compressed_index_path,
    get_decompressed_content_at_line,
    get_decompressed_lines,
    get_or_build_compressed_index,
    is_compressed_index_valid,
    list_compressed_indexes,
    load_compressed_index,
    save_compressed_index,
)


@pytest.fixture
def temp_gzip_file():
    """Create a temporary gzip file with known content."""
    lines = [f'Line {i:04d}: Content for line {i}\n' for i in range(1, 101)]
    content = ''.join(lines).encode('utf-8')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.gz') as f:
        temp_path = f.name

    with gzip.open(temp_path, 'wb') as gz:
        gz.write(content)

    yield temp_path, content, lines

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_gzip_large():
    """Create a larger temporary gzip file for checkpoint testing."""
    lines = [f'Line {i:05d}: Content for line number {i}\n' for i in range(1, 5001)]
    content = ''.join(lines).encode('utf-8')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.gz') as f:
        temp_path = f.name

    with gzip.open(temp_path, 'wb') as gz:
        gz.write(content)

    yield temp_path, content, lines

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_index_dir(tmp_path):
    """Create a temporary index directory."""
    with patch('rx.compressed_index.DEFAULT_CACHE_DIR', tmp_path / 'compressed_indexes'):
        yield tmp_path / 'compressed_indexes'


@pytest.fixture
def cleanup_indexes(temp_gzip_file):
    """Cleanup any created indexes after test."""
    temp_path, _, _ = temp_gzip_file
    yield temp_path
    delete_compressed_index(temp_path)


class TestIndexPaths:
    """Tests for index path generation."""

    def test_get_compressed_index_dir_creates_directory(self, temp_index_dir):
        """Test that index directory is created."""
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            index_dir = get_compressed_index_dir()
            assert index_dir.exists()
            assert index_dir.is_dir()

    def test_get_compressed_index_path_format(self, temp_gzip_file, temp_index_dir):
        """Test index path format."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            index_path = get_compressed_index_path(temp_path)
            assert index_path.suffix == '.json'
            assert temp_path.split('/')[-1] in str(index_path)

    def test_get_compressed_index_path_consistent(self, temp_gzip_file, temp_index_dir):
        """Test that index path is consistent for same source."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            path1 = get_compressed_index_path(temp_path)
            path2 = get_compressed_index_path(temp_path)
            assert path1 == path2


class TestIndexBuild:
    """Tests for building compressed file indexes."""

    def test_build_compressed_index(self, temp_gzip_file, temp_index_dir):
        """Test building an index for a compressed file."""
        temp_path, content, lines = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            index_data = build_compressed_index(temp_path)

            assert index_data['version'] == INDEX_VERSION
            assert index_data['source_path'] == str(Path(temp_path).resolve())
            assert index_data['compression_format'] == 'gzip'
            assert index_data['decompressed_size_bytes'] == len(content)
            # Line count may be off by one due to final newline counting
            assert abs(index_data['total_lines'] - len(lines)) <= 1
            assert 'line_index' in index_data
            assert len(index_data['line_index']) > 0

    def test_build_index_has_first_line_checkpoint(self, temp_gzip_file, temp_index_dir):
        """Test that index always has checkpoint for line 1."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            index_data = build_compressed_index(temp_path)
            assert index_data['line_index'][0] == [1, 0]

    def test_build_index_noncompressed_raises(self, temp_index_dir):
        """Test building index for non-compressed file raises ValueError."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
            f.write(b'Plain text content\n')
            temp_path = f.name

        try:
            with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir), pytest.raises(ValueError):
                build_compressed_index(temp_path)
        finally:
            os.unlink(temp_path)


class TestIndexSaveLoad:
    """Tests for saving and loading indexes."""

    def test_save_and_load_index(self, temp_gzip_file, temp_index_dir):
        """Test saving and loading an index."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            index_data = build_compressed_index(temp_path)
            save_compressed_index(index_data, temp_path)

            loaded = load_compressed_index(temp_path)
            assert loaded is not None
            assert loaded['version'] == index_data['version']
            assert loaded['source_path'] == index_data['source_path']

    def test_load_nonexistent_index(self, temp_gzip_file, temp_index_dir):
        """Test loading non-existent index returns None."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            loaded = load_compressed_index(temp_path)
            assert loaded is None


class TestIndexValidation:
    """Tests for index validation."""

    def test_is_valid_with_fresh_index(self, temp_gzip_file, temp_index_dir):
        """Test validation passes for fresh index."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            index_data = build_compressed_index(temp_path)
            save_compressed_index(index_data, temp_path)

            assert is_compressed_index_valid(temp_path) is True

    def test_is_invalid_with_no_index(self, temp_gzip_file, temp_index_dir):
        """Test validation fails when no index exists."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            assert is_compressed_index_valid(temp_path) is False

    def test_is_invalid_with_wrong_version(self, temp_gzip_file, temp_index_dir):
        """Test validation fails with wrong version."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            index_data = build_compressed_index(temp_path)
            index_data['version'] = INDEX_VERSION + 1
            save_compressed_index(index_data, temp_path)

            assert is_compressed_index_valid(temp_path) is False

    def test_is_invalid_with_changed_size(self, temp_gzip_file, temp_index_dir):
        """Test validation fails when source file size changes."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            index_data = build_compressed_index(temp_path)
            index_data['source_size_bytes'] += 1000
            save_compressed_index(index_data, temp_path)

            assert is_compressed_index_valid(temp_path) is False


class TestGetOrBuildIndex:
    """Tests for get_or_build_compressed_index."""

    def test_builds_new_index(self, temp_gzip_file, temp_index_dir):
        """Test building new index when none exists."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            index_data = get_or_build_compressed_index(temp_path)
            assert index_data is not None
            assert 'version' in index_data

    def test_uses_cached_index(self, temp_gzip_file, temp_index_dir):
        """Test using cached index when valid."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            # Build and save
            index_data1 = get_or_build_compressed_index(temp_path)

            # Should use cached version
            index_data2 = get_or_build_compressed_index(temp_path)

            assert index_data1['created_at'] == index_data2['created_at']


class TestFindNearestCheckpoint:
    """Tests for checkpoint lookup."""

    def test_find_checkpoint_at_start(self):
        """Test finding checkpoint at start of file."""
        index_data = {'line_index': [[1, 0], [1000, 50000], [2000, 100000]]}
        line, offset = find_nearest_checkpoint(index_data, 1)
        assert line == 1
        assert offset == 0

    def test_find_checkpoint_in_middle(self):
        """Test finding checkpoint in middle of file."""
        index_data = {'line_index': [[1, 0], [1000, 50000], [2000, 100000]]}
        line, offset = find_nearest_checkpoint(index_data, 1500)
        assert line == 1000
        assert offset == 50000

    def test_find_checkpoint_at_exact_boundary(self):
        """Test finding checkpoint at exact checkpoint line."""
        index_data = {'line_index': [[1, 0], [1000, 50000], [2000, 100000]]}
        line, offset = find_nearest_checkpoint(index_data, 2000)
        assert line == 2000
        assert offset == 100000

    def test_find_checkpoint_past_end(self):
        """Test finding checkpoint past last checkpoint."""
        index_data = {'line_index': [[1, 0], [1000, 50000], [2000, 100000]]}
        line, offset = find_nearest_checkpoint(index_data, 5000)
        assert line == 2000
        assert offset == 100000


class TestGetDecompressedLines:
    """Tests for retrieving specific lines from compressed files."""

    def test_get_first_line(self, temp_gzip_file, temp_index_dir):
        """Test getting the first line."""
        temp_path, _, lines = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            result = get_decompressed_lines(temp_path, 1, 1)
            assert len(result) == 1
            assert result[0] == lines[0].rstrip('\n')

    def test_get_multiple_lines(self, temp_gzip_file, temp_index_dir):
        """Test getting multiple lines."""
        temp_path, _, lines = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            result = get_decompressed_lines(temp_path, 10, 5)
            assert len(result) == 5
            for i, line in enumerate(result):
                assert line == lines[9 + i].rstrip('\n')

    def test_get_lines_near_end(self, temp_gzip_file, temp_index_dir):
        """Test getting lines near end of file."""
        temp_path, _, lines = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            result = get_decompressed_lines(temp_path, 98, 3)
            assert len(result) == 3
            assert result[0] == lines[97].rstrip('\n')


class TestGetDecompressedContentAtLine:
    """Tests for getting content with context around a line."""

    def test_get_content_with_context(self, temp_gzip_file, temp_index_dir):
        """Test getting content with context lines."""
        temp_path, _, lines = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            result = get_decompressed_content_at_line(temp_path, 50, context_before=2, context_after=2)
            assert len(result) == 5  # 2 before + 1 target + 2 after
            assert result[2] == lines[49].rstrip('\n')  # Line 50 is at index 49

    def test_get_content_at_start(self, temp_gzip_file, temp_index_dir):
        """Test getting content at start of file."""
        temp_path, _, lines = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            result = get_decompressed_content_at_line(temp_path, 1, context_before=2, context_after=2)
            # Can't have 2 lines before line 1
            assert len(result) == 3  # Line 1, 2, 3
            assert result[0] == lines[0].rstrip('\n')

    def test_get_content_with_no_context(self, temp_gzip_file, temp_index_dir):
        """Test getting content with no context."""
        temp_path, _, lines = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            result = get_decompressed_content_at_line(temp_path, 50, context_before=0, context_after=0)
            assert len(result) == 1
            assert result[0] == lines[49].rstrip('\n')


class TestIndexManagement:
    """Tests for index management functions."""

    def test_delete_index(self, temp_gzip_file, temp_index_dir):
        """Test deleting an index."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            index_data = build_compressed_index(temp_path)
            save_compressed_index(index_data, temp_path)

            assert is_compressed_index_valid(temp_path) is True
            deleted = delete_compressed_index(temp_path)
            assert deleted is True
            assert is_compressed_index_valid(temp_path) is False

    def test_delete_nonexistent_index(self, temp_gzip_file, temp_index_dir):
        """Test deleting non-existent index returns False."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            deleted = delete_compressed_index(temp_path)
            assert deleted is False

    def test_list_indexes(self, temp_gzip_file, temp_index_dir):
        """Test listing all indexes."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            # Create an index
            index_data = get_or_build_compressed_index(temp_path)

            indexes = list_compressed_indexes()
            assert len(indexes) >= 1
            assert any(idx['source_path'] == index_data['source_path'] for idx in indexes)

    def test_clear_indexes(self, temp_gzip_file, temp_index_dir):
        """Test clearing all indexes."""
        temp_path, _, _ = temp_gzip_file
        with patch('rx.compressed_index.DEFAULT_CACHE_DIR', temp_index_dir):
            # Create an index
            get_or_build_compressed_index(temp_path)

            count = clear_compressed_indexes()
            assert count >= 1

            indexes = list_compressed_indexes()
            assert len(indexes) == 0
