"""Unit tests for search.py helper functions"""

import pytest
import click
from io import StringIO
from unittest.mock import Mock, patch

from rx.cli.search import (
    format_context_header,
    find_match_for_context,
    build_lines_dict,
    highlight_pattern_in_line,
    display_context_block,
    display_samples_output,
    handle_samples_output,
)
from rx.models import TraceResponse, Match, ContextLine


class TestFormatContextHeader:
    """Tests for format_context_header()"""

    def test_format_without_color(self):
        """Test header formatting without colors"""
        result = format_context_header(
            file_val="/path/to/file.txt", offset_str="123", pattern_val="error", colorize=False
        )
        assert result == "=== /path/to/file.txt:123 [error] ==="

    def test_format_with_color(self):
        """Test header formatting with colors"""
        result = format_context_header(
            file_val="/path/to/file.txt", offset_str="123", pattern_val="error", colorize=True
        )
        # Should contain ANSI escape codes for colors
        assert "\x1b[" in result  # ANSI escape sequence
        assert "/path/to/file.txt" in result
        assert "123" in result
        assert "error" in result

    def test_format_handles_special_characters(self):
        """Test header formatting with special characters in paths"""
        result = format_context_header(
            file_val="/path/with spaces/file (1).txt", offset_str="999", pattern_val="[a-z]+", colorize=False
        )
        assert result == "=== /path/with spaces/file (1).txt:999 [[a-z]+] ==="


class TestFindMatchForContext:
    """Tests for find_match_for_context()"""

    def test_finds_matching_line(self):
        """Test finding a match that exists"""
        matches = [
            Match(pattern="p1", file="f1", offset=100, line_number=5, line_text="error found"),
            Match(pattern="p1", file="f1", offset=200, line_number=10, line_text="warning found"),
        ]
        response = TraceResponse(
            path="/test.txt",
            time=0.1,
            patterns={"p1": "error"},
            files={"f1": "/test.txt"},
            matches=matches,
            scanned_files=[],
            skipped_files=[],
        )

        line_text, line_num = find_match_for_context(response, "p1", "f1", 100)

        assert line_text == "error found"
        assert line_num == 5

    def test_returns_none_when_not_found(self):
        """Test returns None when match doesn't exist"""
        response = TraceResponse(
            path="/test.txt",
            time=0.1,
            patterns={"p1": "error"},
            files={"f1": "/test.txt"},
            matches=[],
            scanned_files=[],
            skipped_files=[],
        )

        line_text, line_num = find_match_for_context(response, "p1", "f1", 100)

        assert line_text is None
        assert line_num is None

    def test_finds_correct_match_among_many(self):
        """Test finds the right match when multiple exist"""
        matches = [
            Match(pattern="p1", file="f1", offset=100, line_number=5, line_text="first"),
            Match(pattern="p2", file="f1", offset=100, line_number=5, line_text="second"),
            Match(pattern="p1", file="f2", offset=100, line_number=5, line_text="third"),
            Match(pattern="p1", file="f1", offset=200, line_number=10, line_text="fourth"),
        ]
        response = TraceResponse(
            path="/test.txt",
            time=0.1,
            patterns={"p1": "error", "p2": "warning"},
            files={"f1": "/test1.txt", "f2": "/test2.txt"},
            matches=matches,
            scanned_files=[],
            skipped_files=[],
        )

        # Should find first match with exact pattern_id, file_id, offset combination
        line_text, line_num = find_match_for_context(response, "p1", "f1", 100)
        assert line_text == "first"


class TestBuildLinesDict:
    """Tests for build_lines_dict()"""

    def test_builds_dict_from_context_lines(self):
        """Test building dict from ContextLine objects"""
        ctx_lines = [
            ContextLine(line_number=5, line_text="line 5", absolute_offset=100),
            ContextLine(line_number=6, line_text="line 6", absolute_offset=120),
            ContextLine(line_number=7, line_text="line 7", absolute_offset=140),
        ]

        result = build_lines_dict(ctx_lines, None, None)

        assert result == {5: "line 5", 6: "line 6", 7: "line 7"}

    def test_adds_matched_line(self):
        """Test that matched line is added to dict"""
        ctx_lines = [
            ContextLine(line_number=5, line_text="before", absolute_offset=100),
            ContextLine(line_number=7, line_text="after", absolute_offset=140),
        ]

        result = build_lines_dict(ctx_lines, "matched line", 6)

        assert result == {5: "before", 6: "matched line", 7: "after"}

    def test_handles_dict_context_lines(self):
        """Test building dict from dict-style context lines"""
        ctx_lines = [
            {"line_number": 5, "line_text": "line 5"},
            {"line_number": 6, "line_text": "line 6"},
        ]

        result = build_lines_dict(ctx_lines, None, None)

        assert result == {5: "line 5", 6: "line 6"}

    def test_handles_empty_context_lines(self):
        """Test with no context lines, only matched line"""
        result = build_lines_dict([], "only match", 10)

        assert result == {10: "only match"}

    def test_matched_line_overrides_context(self):
        """Test that matched line takes precedence if line number conflicts"""
        ctx_lines = [
            ContextLine(line_number=5, line_text="context version", absolute_offset=100),
        ]

        result = build_lines_dict(ctx_lines, "matched version", 5)

        # Matched line should override context line at same line number
        assert result[5] == "matched version"


class TestHighlightPatternInLine:
    """Tests for highlight_pattern_in_line()"""

    def test_returns_unchanged_when_no_color(self):
        """Test returns original line when colorize is False"""
        result = highlight_pattern_in_line("error found here", "error", colorize=False)
        assert result == "error found here"

    def test_highlights_pattern_with_color(self):
        """Test adds color codes when colorize is True"""
        result = highlight_pattern_in_line("error found here", "error", colorize=True)

        # Should contain ANSI escape codes
        assert "\x1b[" in result
        assert "error" in result
        assert "found here" in result

    def test_handles_multiple_occurrences(self):
        """Test highlights multiple pattern occurrences"""
        result = highlight_pattern_in_line("error and error again", "error", colorize=True)

        # Both occurrences should be highlighted
        assert result.count("\x1b[") >= 2  # At least 2 color codes

    def test_handles_regex_patterns(self):
        """Test handles regex patterns correctly"""
        result = highlight_pattern_in_line("error123 and error456", r"error\d+", colorize=True)

        # Should highlight regex matches
        assert "\x1b[" in result

    def test_handles_invalid_regex_gracefully(self):
        """Test returns original line if regex is invalid"""
        result = highlight_pattern_in_line("some text", "[invalid(regex", colorize=True)

        # Should return original line without crashing
        assert result == "some text"

    def test_handles_special_regex_characters(self):
        """Test handles special characters that need escaping"""
        result = highlight_pattern_in_line("find [this] here", r"\[this\]", colorize=True)

        # Should highlight the bracketed text
        assert "\x1b[" in result or result == "find [this] here"  # Graceful fallback


class TestDisplayContextBlock:
    """Tests for display_context_block()"""

    @patch('click.echo')
    def test_displays_complete_context_block(self, mock_echo):
        """Test displays header and context lines"""
        matches = [
            Match(pattern="p1", file="f1", offset=100, line_number=5, line_text="matched line"),
        ]
        context_lines = {
            "p1:f1:100": [
                ContextLine(line_number=4, line_text="before", absolute_offset=80),
                ContextLine(line_number=6, line_text="after", absolute_offset=120),
            ]
        }
        response = TraceResponse(
            path="/test.txt",
            time=0.1,
            patterns={"p1": "error"},
            files={"f1": "/test.txt"},
            matches=matches,
            context_lines=context_lines,
            scanned_files=[],
            skipped_files=[],
        )

        display_context_block("p1:f1:100", response, {"p1": "error"}, {"f1": "/test.txt"}, colorize=False)

        # Should have called echo for: header, before line, matched line, after line, blank line
        assert mock_echo.call_count >= 5

        # Verify header was displayed
        header_call = mock_echo.call_args_list[0][0][0]
        assert "/test.txt" in header_call
        assert "100" in header_call
        assert "error" in header_call

    @patch('click.echo')
    def test_handles_invalid_composite_key(self, mock_echo):
        """Test handles malformed composite key gracefully"""
        response = TraceResponse(
            path="/test.txt",
            time=0.1,
            patterns={"p1": "error"},
            files={"f1": "/test.txt"},
            matches=[],
            scanned_files=[],
            skipped_files=[],
        )

        # Invalid key format (only 2 parts instead of 3)
        display_context_block(
            "p1:f1",  # Missing offset
            response,
            {"p1": "error"},
            {"f1": "/test.txt"},
            colorize=False,
        )

        # Should not crash, should not display anything
        mock_echo.assert_not_called()


class TestDisplaySamplesOutput:
    """Tests for display_samples_output()"""

    @patch('click.echo')
    def test_displays_samples_header(self, mock_echo):
        """Test displays header with context info"""
        response = TraceResponse(
            path="/test.txt",
            time=0.1,
            patterns={"p1": "error"},
            files={"f1": "/test.txt"},
            matches=[],
            scanned_files=[],
            skipped_files=[],
        )

        display_samples_output(response, {"p1": "error"}, {"f1": "/test.txt"}, 3, 5, colorize=False)

        # Should display "Samples (context: 3 before, 5 after):"
        calls = [str(call) for call in mock_echo.call_args_list]
        assert any("Samples (context: 3 before, 5 after):" in str(call) for call in calls)

    @patch('click.echo')
    def test_displays_no_context_message_when_empty(self, mock_echo):
        """Test shows message when no context available"""
        response = TraceResponse(
            path="/test.txt",
            time=0.1,
            patterns={"p1": "error"},
            files={"f1": "/test.txt"},
            matches=[],
            context_lines=None,  # No context
            scanned_files=[],
            skipped_files=[],
        )

        display_samples_output(response, {"p1": "error"}, {"f1": "/test.txt"}, 3, 3, colorize=False)

        # Should show "No context available" message
        calls = [str(call) for call in mock_echo.call_args_list]
        assert any("No context available" in str(call) for call in calls)


class TestHandleSamplesOutput:
    """Tests for handle_samples_output()"""

    @patch('click.echo')
    def test_outputs_json_when_requested(self, mock_echo):
        """Test outputs JSON format when output_json=True"""
        matches = [
            Match(pattern="p1", file="f1", offset=100, line_number=5, line_text="error found"),
        ]
        response = TraceResponse(
            path="/test.txt",
            time=0.1,
            patterns={"p1": "error"},
            files={"f1": "/test.txt"},
            matches=matches,
            scanned_files=[],
            skipped_files=[],
        )

        handle_samples_output(response, {"p1": "error"}, {"f1": "/test.txt"}, 3, 3, output_json=True, no_color=False)

        # Should output JSON
        assert mock_echo.call_count == 1
        output = mock_echo.call_args[0][0]
        assert output.startswith("{")  # JSON object
        assert '"path"' in output
        assert '"matches"' in output

    @patch('rx.cli.search.display_samples_output')
    def test_calls_display_function_for_cli_output(self, mock_display):
        """Test calls display_samples_output for CLI format"""
        response = TraceResponse(
            path="/test.txt",
            time=0.1,
            patterns={"p1": "error"},
            files={"f1": "/test.txt"},
            matches=[],
            scanned_files=[],
            skipped_files=[],
        )

        handle_samples_output(response, {"p1": "error"}, {"f1": "/test.txt"}, 3, 3, output_json=False, no_color=True)

        # Should call display_samples_output
        mock_display.assert_called_once()
        call_args = mock_display.call_args
        # Parameters: response, pattern_ids, file_ids, before_ctx, after_ctx, colorize
        assert call_args[0][0] == response
        assert call_args[0][1] == {"p1": "error"}
        assert call_args[0][2] == {"f1": "/test.txt"}
        assert call_args[0][3] == 3  # before_ctx
        assert call_args[0][4] == 3  # after_ctx
        assert call_args[0][5] == False  # colorize should be False when no_color=True

    @patch('click.echo')
    @patch('sys.exit')
    def test_handles_errors_gracefully(self, mock_exit, mock_echo):
        """Test handles exceptions and exits with error"""
        # Create response that will cause error when dumping to JSON
        response = Mock()
        response.model_dump.side_effect = Exception("Test error")

        handle_samples_output(response, {}, {}, 3, 3, output_json=True, no_color=False)

        # Should display error message
        calls = [str(call) for call in mock_echo.call_args_list]
        assert any("Error displaying samples" in str(call) for call in calls)

        # Should exit with error code
        mock_exit.assert_called_once_with(1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
