"""Compression format detection and decompression utilities.

This module provides functionality to detect compression formats, get
appropriate decompressor commands, and manage compressed file operations.

Supported formats:
- gzip (.gz)
- zstandard (.zst)
- xz (.xz)
- bzip2 (.bz2)
"""

import shutil
import subprocess
from enum import Enum
from pathlib import Path


# Magic bytes for compression format detection
MAGIC_BYTES = {
    b'\x1f\x8b': 'gzip',  # gzip magic
    b'\x28\xb5\x2f\xfd': 'zstd',  # zstd magic
    b'\xfd\x37\x7a\x58\x5a\x00': 'xz',  # xz magic
    b'\x42\x5a\x68': 'bz2',  # bzip2 magic "BZh"
}

# Extension to format mapping
EXTENSION_MAP = {
    '.gz': 'gzip',
    '.gzip': 'gzip',
    '.zst': 'zstd',
    '.zstd': 'zstd',
    '.xz': 'xz',
    '.bz2': 'bz2',
    '.bzip2': 'bz2',
}

# Compound archive extensions that should NOT be treated as simple compressed files
# These contain archives (tar, etc.) inside and require special handling
COMPOUND_ARCHIVE_SUFFIXES = {
    '.tar.gz',
    '.tgz',
    '.tar.zst',
    '.tzst',
    '.tar.xz',
    '.txz',
    '.tar.bz2',
    '.tbz2',
    '.tbz',
}

# Decompressor commands for each format
# Each returns a command that reads from file and writes to stdout
DECOMPRESSOR_COMMANDS = {
    'gzip': ['gzip', '-d', '-c'],
    'zstd': ['zstd', '-d', '-c', '-q'],  # -q for quiet
    'xz': ['xz', '-d', '-c'],
    'bz2': ['bzip2', '-d', '-c'],
}

# Alternative decompressor commands (cat variants)
DECOMPRESSOR_COMMANDS_ALT = {
    'gzip': ['zcat'],
    'zstd': ['zstdcat'],
    'xz': ['xzcat'],
    'bz2': ['bzcat'],
}


class CompressionFormat(Enum):
    """Supported compression formats."""

    NONE = 'none'
    GZIP = 'gzip'
    ZSTD = 'zstd'
    XZ = 'xz'
    BZ2 = 'bz2'

    @classmethod
    def from_string(cls, value: str) -> 'CompressionFormat':
        """Create CompressionFormat from string value."""
        for member in cls:
            if member.value == value:
                return member
        return cls.NONE


def is_compound_archive(filepath: str | Path) -> bool:
    """Check if file is a compound archive (e.g., .tar.gz) that we don't support.

    Compound archives contain another archive format inside the compression,
    so decompressing them doesn't yield searchable text content.

    Args:
        filepath: Path to the file

    Returns:
        True if file is a compound archive, False otherwise
    """
    name = Path(filepath).name.lower()
    for suffix in COMPOUND_ARCHIVE_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def detect_compression_by_extension(filepath: str | Path) -> CompressionFormat:
    """Detect compression format by file extension.

    Args:
        filepath: Path to the file

    Returns:
        Detected CompressionFormat, or NONE if not recognized or compound archive
    """
    # Skip compound archives like .tar.gz
    if is_compound_archive(filepath):
        return CompressionFormat.NONE

    path = Path(filepath)
    suffix = path.suffix.lower()

    format_str = EXTENSION_MAP.get(suffix)
    if format_str:
        return CompressionFormat.from_string(format_str)

    return CompressionFormat.NONE


def detect_compression_by_magic(filepath: str | Path) -> CompressionFormat:
    """Detect compression format by reading magic bytes.

    Args:
        filepath: Path to the file

    Returns:
        Detected CompressionFormat, or NONE if not recognized or compound archive
    """
    # Skip compound archives like .tar.gz
    if is_compound_archive(filepath):
        return CompressionFormat.NONE

    try:
        with open(filepath, 'rb') as f:
            header = f.read(6)  # Read enough for longest magic

        for magic, format_str in MAGIC_BYTES.items():
            if header.startswith(magic):
                return CompressionFormat.from_string(format_str)

    except OSError:
        pass

    return CompressionFormat.NONE


def detect_compression(filepath: str | Path) -> CompressionFormat:
    """Detect compression format by extension first, then magic bytes.

    Args:
        filepath: Path to the file

    Returns:
        Detected CompressionFormat, or NONE if not compressed
    """
    # Try extension first (faster)
    format_by_ext = detect_compression_by_extension(filepath)
    if format_by_ext != CompressionFormat.NONE:
        return format_by_ext

    # Fall back to magic bytes
    return detect_compression_by_magic(filepath)


def is_compressed(filepath: str | Path) -> bool:
    """Check if a file is compressed.

    Args:
        filepath: Path to the file

    Returns:
        True if file is compressed, False otherwise
    """
    return detect_compression(filepath) != CompressionFormat.NONE


def get_decompressor_command(format: CompressionFormat, filepath: str | Path | None = None) -> list[str]:
    """Get the command to decompress a file to stdout.

    Args:
        format: Compression format
        filepath: Optional file path to append to command

    Returns:
        Command list suitable for subprocess

    Raises:
        ValueError: If format is not supported or NONE
    """
    if format == CompressionFormat.NONE:
        raise ValueError('Cannot get decompressor for uncompressed files')

    cmd = DECOMPRESSOR_COMMANDS.get(format.value)
    if not cmd:
        raise ValueError(f'Unknown compression format: {format}')

    result = cmd.copy()
    if filepath:
        result.append(str(filepath))

    return result


def check_decompressor_available(format: CompressionFormat) -> bool:
    """Check if the decompressor for a format is available.

    Args:
        format: Compression format to check

    Returns:
        True if decompressor is available, False otherwise
    """
    if format == CompressionFormat.NONE:
        return True

    cmd = DECOMPRESSOR_COMMANDS.get(format.value)
    if not cmd:
        return False

    # Check if the command exists
    return shutil.which(cmd[0]) is not None


def get_available_decompressors() -> dict[str, bool]:
    """Get availability status of all decompressors.

    Returns:
        Dict mapping format name to availability boolean
    """
    return {
        format.value: check_decompressor_available(format)
        for format in CompressionFormat
        if format != CompressionFormat.NONE
    }


def get_decompressed_size(filepath: str | Path, format: CompressionFormat) -> int | None:
    """Try to get the decompressed size of a compressed file.

    For gzip files, the uncompressed size is stored in the last 4 bytes.
    For other formats, this requires full decompression.

    Args:
        filepath: Path to the compressed file
        format: Compression format

    Returns:
        Decompressed size in bytes, or None if cannot determine
    """
    if format == CompressionFormat.GZIP:
        try:
            with open(filepath, 'rb') as f:
                f.seek(-4, 2)  # Seek to last 4 bytes
                size_bytes = f.read(4)
                # gzip stores size as little-endian 32-bit integer (mod 2^32)
                return int.from_bytes(size_bytes, 'little')
        except OSError:
            pass

    # For other formats, we can't easily determine size without decompressing
    return None


def decompress_to_stdout(
    filepath: str | Path,
    format: CompressionFormat | None = None,
) -> subprocess.Popen:
    """Start a decompression process that writes to stdout.

    Args:
        filepath: Path to the compressed file
        format: Compression format (auto-detected if not provided)

    Returns:
        Popen object with stdout as PIPE

    Raises:
        ValueError: If format cannot be detected or is not supported
        FileNotFoundError: If decompressor is not available
    """
    if format is None:
        format = detect_compression(filepath)

    if format == CompressionFormat.NONE:
        raise ValueError(f'File is not compressed or format not recognized: {filepath}')

    if not check_decompressor_available(format):
        raise FileNotFoundError(
            f'Decompressor for {format.value} not found. Please install: {DECOMPRESSOR_COMMANDS[format.value][0]}'
        )

    cmd = get_decompressor_command(format, filepath)

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def decompress_file(
    filepath: str | Path,
    format: CompressionFormat | None = None,
) -> bytes:
    """Decompress an entire file and return its contents.

    Warning: This loads the entire decompressed content into memory.
    Use decompress_to_stdout() for large files.

    Args:
        filepath: Path to the compressed file
        format: Compression format (auto-detected if not provided)

    Returns:
        Decompressed file contents as bytes

    Raises:
        ValueError: If format cannot be detected or is not supported
        RuntimeError: If decompression fails
    """
    proc = decompress_to_stdout(filepath, format)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f'Decompression failed with code {proc.returncode}: {stderr.decode()}')

    return stdout


def decompress_to_file(
    input_path: str | Path,
    output_path: str | Path,
    format: CompressionFormat | None = None,
) -> None:
    """Decompress a file to the output path.

    Args:
        input_path: Path to compressed file
        output_path: Path where decompressed file should be written
        format: Compression format (auto-detected if not provided)

    Raises:
        ValueError: If file is not compressed or format unsupported
        FileNotFoundError: If decompressor is not available
        OSError: If decompression fails (e.g., no space left on device)
    """
    if format is None:
        format = detect_compression(input_path)

    if format == CompressionFormat.NONE:
        raise ValueError(f'File is not compressed: {input_path}')

    if not check_decompressor_available(format):
        raise FileNotFoundError(
            f'Decompressor for {format.value} not found. Please install: {DECOMPRESSOR_COMMANDS[format.value][0]}'
        )

    cmd = get_decompressor_command(format, input_path)

    # Run decompression to file
    try:
        with open(output_path, 'wb') as outfile:
            result = subprocess.run(
                cmd,
                stdout=outfile,
                stderr=subprocess.PIPE,
                check=False,
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='replace')
                raise OSError(f'Decompression failed: {error_msg}')
    except OSError:
        # Re-raise OSError (includes "No space left" errors)
        raise
    except Exception as e:
        # Wrap other exceptions as OSError
        raise OSError(f'Decompression failed: {e}') from e
