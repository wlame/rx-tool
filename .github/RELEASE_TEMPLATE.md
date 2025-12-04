<!-- Copy this template when creating a new release -->

## What's Changed

### Added
- 

### Changed
- 

### Fixed
- 

### Security
- 

## Installation

Download the binary for your platform:

- **Linux**: `rx-linux-x86_64`
- **Windows**: `rx-windows-x86_64.exe`
- **macOS**: `rx-macos-x86_64`

### Verify Download (Optional)

```bash
# Download checksums.txt from this release
# Then verify your binary:
sha256sum -c checksums.txt --ignore-missing
```

### Linux/macOS

```bash
# Make executable
chmod +x rx-linux-x86_64  # or rx-macos-x86_64

# Run
./rx-linux-x86_64 --version

# Optional: Move to PATH
sudo mv rx-linux-x86_64 /usr/local/bin/rx
```

### Windows

```bash
# Run
rx-windows-x86_64.exe --version

# Optional: Add to PATH or rename to rx.exe
```

## Full Changelog

See [CHANGELOG.md](https://github.com/wlame/rx-tool/blob/main/CHANGELOG.md) for complete details.

**Full Changelog**: https://github.com/wlame/rx-tool/compare/v0.x.x...v1.0.0
