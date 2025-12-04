# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-12-03

### Added
- **Search Root Security**: `--search-root` parameter for `rx serve` command to restrict file access within a specific directory
  - Path traversal attack prevention (blocks `../` escapes)
  - Symlink resolution and validation
  - Configurable via CLI parameter or `RX_SEARCH_ROOT` environment variable
  - Returns HTTP 403 Forbidden for paths outside search root
  - Search root displayed in `/health` endpoint
- **Webhook Support**: HTTP webhooks for trace events
  - `on_file_scanned`: Called when file scan completes
  - `on_match_found`: Called for each match found
  - `on_trace_complete`: Called when trace request finishes
  - Configurable via CLI options or environment variables
  - Non-blocking async hook calls with 3-second timeout
  - Request tracking with UUID v7 request IDs
- Comprehensive test suite for path security (30 tests)
- Request store for tracking active requests in serve mode

### Changed
- Documentation examples now use `=` for long-form CLI parameters (e.g., `--port=8000`)
- Updated `/health` endpoint to include search root information
- Added request_id field to TraceResponse model

### Security
- Implemented security sandbox for file operations in serve mode
- Protection against directory traversal attacks
- Protection against symlink escape attacks

## [1.0.0] - 2025-12-01

### Added
- Line offset support for `rx samples` command with `-l/--line-offset` option
- Large file indexing system for efficient line-based access
- `rx index` CLI command for managing large file indexes
- `rx samples` CLI command for viewing file content around byte offsets or line numbers
- `/v1/samples` API endpoint with support for both byte offsets and line numbers
- File analysis features: p95/p99 percentiles, longest line info, line ending detection
- Favicon support in web UI
- Comprehensive test suite (321 tests)

### Changed
- Renamed analysis fields from `*_line_length` to `line_length_*` for consistency
- Directory analysis now skips binary files and processes in parallel
- Updated to require Python 3.13+

### Fixed
- Context output formatting issues
- Line number tracking in search results
- Binary file detection and handling

## [Unreleased]

### Planned
- Additional export formats for analysis results
- Performance optimizations for very large files
- Enhanced regex complexity analysis

---

## Release Process

To create a new release:

1. Update version in `src/rx/__version__.py` and `pyproject.toml`
2. Update `CHANGELOG.md` with changes for the new version
3. Commit changes: `git commit -am "Release v1.x.x"`
4. Create and push tag: `git tag v1.x.x && git push origin v1.x.x`
5. Create a release on GitHub using the tag
6. GitHub Actions will automatically build binaries for all platforms and attach them to the release

[1.1.0]: https://github.com/wlame/rx-tool/releases/tag/v1.1.0
[1.0.0]: https://github.com/wlame/rx-tool/releases/tag/v1.0.0
