"""Tests for parse module functions and regex complexity analysis"""

import os
import tempfile

import pytest

from rx.file_utils import (
    get_context,
    get_file_offsets,
)
from rx.regex import calculate_regex_complexity


class TestGetFileOffsets:
    """Tests for get_file_offsets function"""

    def _create_test_file(self, size_mb: int) -> str:
        """Helper to create a test file of specified size"""
        # Create file with lines to ensure proper newline alignment
        f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        line = "A" * 100 + "\n"  # 101 bytes per line
        lines_needed = (size_mb * 1024 * 1024) // len(line)
        for _ in range(lines_needed):
            f.write(line)
        f.close()
        return f.name

    def test_small_file_single_chunk(self):
        """Files smaller than MIN_CHUNK_SIZE should return single offset [0]"""
        filepath = self._create_test_file(10)  # 10MB
        try:
            offsets = get_file_offsets(filepath, os.path.getsize(filepath))
            assert offsets == [0]
        finally:
            os.unlink(filepath)

    def test_120mb_file_two_chunks(self):
        """120MB file should be split into 5 chunks (25MB threshold)"""
        filepath = self._create_test_file(120)  # 120MB
        try:
            offsets = get_file_offsets(filepath, os.path.getsize(filepath))
            # Should have 5 chunks (120MB / 25MB = 4.8, rounded up to 5)
            assert len(offsets) == 5
            # First offset always 0
            assert offsets[0] == 0
            # Second offset should be around 25MB (aligned to newline)
            assert 20 * 1024 * 1024 < offsets[1] < 30 * 1024 * 1024
        finally:
            os.unlink(filepath)

    def test_offsets_always_start_with_zero(self):
        """All offset lists should start with 0"""
        for size_mb in [10, 100, 200]:
            filepath = self._create_test_file(size_mb)
            try:
                offsets = get_file_offsets(filepath, os.path.getsize(filepath))
                assert offsets[0] == 0
            finally:
                os.unlink(filepath)

    def test_offsets_aligned_to_newlines(self):
        """Offsets should be aligned to newline boundaries"""
        filepath = self._create_test_file(150)  # 150MB -> should split into 3 chunks
        try:
            offsets = get_file_offsets(filepath, os.path.getsize(filepath))

            # Read file to verify offsets are at newline boundaries
            with open(filepath, 'rb') as f:
                for i, offset in enumerate(offsets):
                    if offset == 0:
                        continue
                    # Check that the byte before offset is a newline
                    f.seek(offset - 1)
                    byte_before = f.read(1)
                    assert byte_before == b'\n', f"Offset {i} ({offset}) is not aligned to newline"
        finally:
            os.unlink(filepath)


class TestRegexComplexity:
    """Tests for calculate_regex_complexity function using AST-based analysis"""

    # =========================================================================
    # Basic functionality tests
    # =========================================================================

    def test_returns_expected_keys(self):
        """Test that result contains all expected keys"""
        result = calculate_regex_complexity('hello')

        # New primary fields
        assert 'score' in result
        assert 'risk_level' in result
        assert 'complexity_class' in result
        assert 'complexity_notation' in result
        assert 'issues' in result
        assert 'recommendations' in result
        assert 'performance' in result
        assert 'star_height' in result
        assert 'pattern_length' in result
        assert 'has_anchors' in result

        # Legacy fields for compatibility
        assert 'level' in result
        assert 'risk' in result
        assert 'warnings' in result
        assert 'details' in result

    def test_score_is_numeric(self):
        """Test that score is numeric"""
        result = calculate_regex_complexity('test.*')
        assert isinstance(result['score'], (int, float))
        assert 0 <= result['score'] <= 100

    def test_issues_is_list(self):
        """Test that issues is always a list"""
        result = calculate_regex_complexity('hello')
        assert isinstance(result['issues'], list)

    def test_invalid_regex_returns_error(self):
        """Test that invalid regex returns error state"""
        result = calculate_regex_complexity('(unclosed')
        assert result['risk_level'] == 'error'
        assert 'parse_error' in result['issues'][0]['type']

    # =========================================================================
    # Linear complexity patterns (safe)
    # =========================================================================

    def test_simple_literal(self):
        """Test literal string (safest pattern)"""
        result = calculate_regex_complexity('hello')
        assert result['risk_level'] == 'safe'
        assert result['complexity_class'] == 'linear'
        assert result['complexity_notation'] == 'O(n)'
        assert len(result['issues']) == 0

    def test_anchored_pattern(self):
        """Test anchored pattern with character class"""
        result = calculate_regex_complexity('^[a-z]+$')
        assert result['risk_level'] == 'safe'
        assert result['complexity_class'] == 'linear'
        assert result['has_anchors'] == (True, True)

    def test_simple_email_pattern(self):
        """Test simple email-like pattern"""
        result = calculate_regex_complexity(r'\w+@\w+\.\w+')
        assert result['risk_level'] in ['safe', 'caution']
        assert result['complexity_class'] == 'linear'

    def test_word_boundary_pattern(self):
        """Test pattern with word boundaries"""
        result = calculate_regex_complexity(r'\bword\b')
        assert result['risk_level'] == 'safe'
        assert result['complexity_class'] == 'linear'

    # =========================================================================
    # Exponential complexity patterns (CRITICAL)
    # =========================================================================

    def test_nested_quantifiers_plus_plus(self):
        """Test (a+)+ - classic nested quantifier ReDoS pattern"""
        result = calculate_regex_complexity('(a+)+')
        assert result['risk_level'] == 'critical'
        assert result['complexity_class'] == 'exponential'
        assert result['complexity_notation'] == 'O(2^n)'
        assert any(i['type'] == 'nested_quantifier' for i in result['issues'])
        assert result['star_height'] >= 2

    def test_nested_quantifiers_star_star(self):
        """Test (a*)* - nested star quantifiers"""
        result = calculate_regex_complexity('(a*)*')
        assert result['risk_level'] == 'critical'
        assert result['complexity_class'] == 'exponential'
        assert any(i['type'] == 'nested_quantifier' for i in result['issues'])

    def test_nested_quantifiers_triple(self):
        """Test ((a+)+)+ - deeply nested quantifiers"""
        result = calculate_regex_complexity('((a+)+)+')
        assert result['risk_level'] == 'critical'
        assert result['complexity_class'] == 'exponential'
        assert result['star_height'] >= 3

    def test_overlapping_alternation(self):
        """Test (a|a)+ - overlapping disjunction"""
        result = calculate_regex_complexity('(a|a)+')
        assert result['risk_level'] == 'critical'
        assert result['complexity_class'] == 'exponential'
        assert any(i['type'] == 'overlapping_disjunction' for i in result['issues'])

    def test_overlapping_alternation_partial(self):
        """Test (a|ab)+ - partial overlap in alternation"""
        result = calculate_regex_complexity('(a|ab)+')
        assert result['risk_level'] == 'critical'
        assert result['complexity_class'] == 'exponential'

    def test_digit_word_overlap(self):
        """Test (\\d|\\w)+ - Python optimizes this to [\\d\\w]+ which is safe"""
        # Python's sre_parse optimizes (\\d|\\w)+ to a character class [\\d\\w]+
        # Character classes don't cause backtracking, so this is actually safe
        result = calculate_regex_complexity(r'(\d|\w)+')
        # This is optimized to a character class, so it's safe
        assert result['complexity_class'] == 'linear'

    # =========================================================================
    # Polynomial complexity patterns (WARNING)
    # =========================================================================

    def test_two_greedy_any(self):
        """Test .*.* - two greedy any-char quantifiers (O(n²))"""
        result = calculate_regex_complexity('.*.*')
        assert result['risk_level'] in ['warning', 'dangerous']
        assert result['complexity_class'] == 'polynomial'
        assert 'n^2' in result['complexity_notation'] or 'n²' in result['complexity_notation']
        assert any(i['type'] == 'greedy_chain' for i in result['issues'])

    def test_three_greedy_any(self):
        """Test .*.*.* - three greedy quantifiers (O(n³))"""
        result = calculate_regex_complexity('.*.*.*')
        assert result['risk_level'] in ['warning', 'dangerous']
        assert result['complexity_class'] == 'polynomial'
        assert '3' in result['complexity_notation']

    def test_four_greedy_any(self):
        """Test .*.*.*.* - four greedy quantifiers (O(n⁴))"""
        result = calculate_regex_complexity('.*.*.*.*')
        assert result['risk_level'] in ['dangerous', 'critical']
        assert result['complexity_class'] == 'polynomial'
        assert '4' in result['complexity_notation']

    def test_greedy_with_literals(self):
        """Test .*error.*failed.* - greedy with literal separators"""
        result = calculate_regex_complexity('.*error.*failed.*')
        assert result['complexity_class'] == 'polynomial'
        assert any(i['type'] == 'greedy_chain' for i in result['issues'])

    # =========================================================================
    # Backreferences
    # =========================================================================

    def test_simple_backreference(self):
        """Test (a)\\1 - simple backreference"""
        result = calculate_regex_complexity(r'(a)\1')
        assert any(i['type'] == 'backreference' for i in result['issues'])
        # Backreferences don't cause exponential by themselves
        assert result['complexity_class'] == 'linear'

    def test_multiple_backreferences(self):
        """Test multiple backreferences"""
        result = calculate_regex_complexity(r'(a)(b)\1\2')
        assert any(i['type'] == 'backreference' for i in result['issues'])

    # =========================================================================
    # Lookarounds
    # =========================================================================

    def test_single_lookahead(self):
        """Test simple lookahead"""
        result = calculate_regex_complexity('(?=.*a)test')
        # Single lookahead is generally fine
        assert result['complexity_class'] == 'linear'

    def test_multiple_lookaheads(self):
        """Test multiple lookaheads"""
        result = calculate_regex_complexity('(?=.*a)(?=.*b)(?=.*c)(?=.*d)')
        # Multiple lookaheads add overhead
        assert any(i['type'] == 'multiple_lookaround' for i in result['issues'])

    def test_nested_lookahead(self):
        """Test nested lookahead"""
        result = calculate_regex_complexity('(?=(?=.*a).*b)')
        assert any(i['type'] == 'nested_lookaround' for i in result['issues'])

    # =========================================================================
    # Star height calculation
    # =========================================================================

    def test_star_height_0(self):
        """Test pattern with no quantifiers has star height 0"""
        result = calculate_regex_complexity('abc')
        assert result['star_height'] == 0

    def test_star_height_1(self):
        """Test pattern with single quantifier has star height 1"""
        result = calculate_regex_complexity('a+')
        assert result['star_height'] == 1

    def test_star_height_2(self):
        """Test nested quantifier has star height 2"""
        result = calculate_regex_complexity('(a+)+')
        assert result['star_height'] == 2

    def test_star_height_3(self):
        """Test deeply nested quantifier has star height 3"""
        result = calculate_regex_complexity('((a+)+)+')
        assert result['star_height'] == 3

    # =========================================================================
    # Fix suggestions and recommendations
    # =========================================================================

    def test_nested_quantifier_has_fix_suggestions(self):
        """Test that nested quantifier issues include fix suggestions"""
        result = calculate_regex_complexity('(a+)+')
        nested_issues = [i for i in result['issues'] if i['type'] == 'nested_quantifier']
        assert len(nested_issues) > 0
        assert len(nested_issues[0]['fix_suggestions']) > 0

    def test_greedy_chain_has_fix_suggestions(self):
        """Test that greedy chain issues include fix suggestions"""
        result = calculate_regex_complexity('.*.*')
        greedy_issues = [i for i in result['issues'] if i['type'] == 'greedy_chain']
        assert len(greedy_issues) > 0
        assert len(greedy_issues[0]['fix_suggestions']) > 0

    def test_safe_pattern_has_recommendations(self):
        """Test that safe patterns have positive recommendations"""
        result = calculate_regex_complexity('^test$')
        assert len(result['recommendations']) > 0

    # =========================================================================
    # Performance estimates
    # =========================================================================

    def test_linear_performance_estimates(self):
        """Test performance estimates for linear patterns"""
        result = calculate_regex_complexity('hello')
        perf = result['performance']
        assert perf['ops_at_100'] == 100
        assert perf['ops_at_1000'] == 1000
        assert perf['ops_at_10000'] == 10000
        assert perf['safe_for_large_files'] is True

    def test_polynomial_performance_estimates(self):
        """Test performance estimates for polynomial patterns"""
        result = calculate_regex_complexity('.*.*')
        perf = result['performance']
        # O(n²) means 100² = 10000, 1000² = 1000000
        assert perf['ops_at_100'] == 10000
        assert perf['ops_at_1000'] == 1000000
        assert perf['safe_for_large_files'] is True  # O(n²) is still usable

    def test_exponential_performance_estimates(self):
        """Test performance estimates for exponential patterns"""
        result = calculate_regex_complexity('(a+)+')
        perf = result['performance']
        # Exponential means astronomical numbers
        assert perf['ops_at_100'] > 10**10
        assert perf['safe_for_large_files'] is False

    # =========================================================================
    # Real-world pattern tests
    # =========================================================================

    def test_email_pattern_safe(self):
        """Test well-designed email pattern is safe"""
        result = calculate_regex_complexity(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        assert result['risk_level'] in ['safe', 'caution']
        assert result['complexity_class'] == 'linear'

    def test_ip_address_pattern(self):
        """Test IP address pattern"""
        result = calculate_regex_complexity(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
        assert result['risk_level'] == 'safe'
        assert result['complexity_class'] == 'linear'

    def test_log_line_pattern(self):
        """Test typical log line pattern"""
        result = calculate_regex_complexity(r'^\[\d{4}-\d{2}-\d{2}\]\s+\[ERROR\]\s+.*$')
        assert result['risk_level'] in ['safe', 'caution']

    def test_vulnerable_email_pattern(self):
        """Test ReDoS-vulnerable email pattern"""
        # This pattern has nested quantifiers and is known to be vulnerable
        result = calculate_regex_complexity('([a-zA-Z0-9]+)*@([a-zA-Z0-9]+)*\\.com')
        assert result['risk_level'] in ['warning', 'dangerous', 'critical']
        assert len(result['issues']) > 0

    # =========================================================================
    # Complexity ordering tests
    # =========================================================================

    def test_complexity_ordering(self):
        """Test that more complex patterns have higher scores"""
        simple = calculate_regex_complexity('hello')
        moderate = calculate_regex_complexity('.*test.*')
        complex_pattern = calculate_regex_complexity('.*.*.*')
        critical = calculate_regex_complexity('(a+)+')

        assert simple['score'] < moderate['score']
        assert moderate['score'] < complex_pattern['score']
        assert complex_pattern['score'] < critical['score']

    def test_risk_level_ordering(self):
        """Test that risk levels follow expected ordering"""
        risk_order = ['safe', 'caution', 'warning', 'dangerous', 'critical']

        patterns = [
            ('hello', 'safe'),
            ('^test$', 'safe'),
            ('.*.*', 'warning'),  # or higher
            ('(a+)+', 'critical'),
        ]

        for pattern, min_expected_risk in patterns:
            result = calculate_regex_complexity(pattern)
            result_idx = risk_order.index(result['risk_level'])
            expected_idx = risk_order.index(min_expected_risk)
            assert result_idx >= expected_idx, (
                f"Pattern '{pattern}' has risk {result['risk_level']}, expected at least {min_expected_risk}"
            )

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_empty_pattern(self):
        """Test empty pattern"""
        result = calculate_regex_complexity('')
        assert result['risk_level'] == 'safe'
        assert result['pattern_length'] == 0

    def test_single_char(self):
        """Test single character pattern"""
        result = calculate_regex_complexity('a')
        assert result['risk_level'] == 'safe'

    def test_escaped_special_chars(self):
        """Test escaped special characters"""
        result = calculate_regex_complexity(r'\(\)\+\*')
        assert result['risk_level'] == 'safe'

    def test_unicode_pattern(self):
        """Test unicode in pattern"""
        result = calculate_regex_complexity('héllo wörld')
        assert result['risk_level'] == 'safe'

    def test_complex_char_class(self):
        """Test complex character class"""
        result = calculate_regex_complexity('[a-zA-Z0-9_.-]+')
        assert result['risk_level'] == 'safe'
        assert result['complexity_class'] == 'linear'
