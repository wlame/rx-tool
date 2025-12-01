"""
Regex Complexity Analysis Module

This module provides AST-based analysis of regex patterns to detect ReDoS
(Regular Expression Denial of Service) vulnerabilities and estimate performance
characteristics.

The analysis is based on academic research into regex engine behavior:

References:
- Regexploit (Doyensec): https://github.com/doyensec/regexploit
- "Static Detection of DoS Vulnerabilities in Programs that Use Regular Expressions"
  https://link.springer.com/chapter/10.1007/978-3-662-54580-5_1
- "Catastrophic Backtracking" https://www.regular-expressions.info/catastrophic.html
- "Regular Expression Matching Can Be Simple And Fast" https://swtch.com/~rsc/regexp/regexp1.html

Vulnerability Pattern Taxonomy:

1. EXPONENTIAL O(2^n) Patterns:
   - Nested Quantifiers (NQ): (a+)+, (a*)*
   - Exponential Overlapping Disjunction (EOD): (a|a)+, (a|ab)+
   - Exponential Overlapping Adjacency (EOA): (a+a+)+

2. POLYNOMIAL O(n^k) Patterns:
   - Polynomial Overlapping Adjacency (POA): .*.*  (O(n²)), .*.*.* (O(n³))
   - Star Height > 1 with overlap

3. LINEAR O(n) but slow:
   - Multiple lookarounds
   - Backreferences (technically NP-complete but usually fast in practice)
"""

from __future__ import annotations

import sre_parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Severity(Enum):
    """Issue severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueType(Enum):
    """Types of regex vulnerabilities"""

    NESTED_QUANTIFIER = "nested_quantifier"
    OVERLAPPING_DISJUNCTION = "overlapping_disjunction"
    OVERLAPPING_ADJACENCY = "overlapping_adjacency"
    GREEDY_CHAIN = "greedy_chain"
    BACKREFERENCE = "backreference"
    NESTED_LOOKAROUND = "nested_lookaround"
    MULTIPLE_LOOKAROUND = "multiple_lookaround"
    LARGE_QUANTIFIER = "large_quantifier"


@dataclass
class RegexIssue:
    """
    Represents a detected vulnerability in a regex pattern.

    Attributes:
        type: The category of vulnerability (e.g., NESTED_QUANTIFIER)
        severity: How critical the issue is (CRITICAL, HIGH, MEDIUM, LOW)
        complexity_class: "exponential", "polynomial", or "linear"
        complexity_notation: Big-O notation (e.g., "O(2^n)", "O(n²)")
        location: Start and end position in original pattern (if available)
        segment: The specific part of the pattern causing the issue
        explanation: Human-readable explanation of why this is problematic
        fix_suggestions: List of specific recommendations to fix the issue
    """

    type: IssueType
    severity: Severity
    complexity_class: str
    complexity_notation: str
    location: tuple[int, int] | None = None
    segment: str = ""
    explanation: str = ""
    fix_suggestions: list[str] = field(default_factory=list)


@dataclass
class PerformanceEstimate:
    """
    Estimated operations for different input sizes.

    Based on complexity class:
    - Linear O(n): ops = n
    - Polynomial O(n^k): ops = n^k
    - Exponential O(2^n): ops = 2^n (capped at practical limits)
    """

    ops_at_100: int
    ops_at_1000: int
    ops_at_10000: int
    safe_for_large_files: bool

    @classmethod
    def from_complexity(cls, complexity_class: str, degree: int = 2) -> PerformanceEstimate:
        """Calculate estimates based on complexity class and polynomial degree."""
        if complexity_class == "linear":
            return cls(ops_at_100=100, ops_at_1000=1000, ops_at_10000=10000, safe_for_large_files=True)
        elif complexity_class == "polynomial":
            return cls(
                ops_at_100=100**degree,
                ops_at_1000=1000**degree,
                ops_at_10000=min(10000**degree, 10**15),  # Cap at practical limit
                safe_for_large_files=degree <= 2,
            )
        else:  # exponential
            return cls(
                ops_at_100=2**50,  # Already astronomical at n=50
                ops_at_1000=2**100,  # Effectively infinite
                ops_at_10000=2**100,
                safe_for_large_files=False,
            )


class CharacterSetAnalyzer:
    """
    Computes what characters a regex element can match.

    Used for detecting overlapping patterns that cause backtracking.
    Returns either a set of specific characters or the special value "ANY"
    to indicate the pattern can match any character.
    """

    # Common category mappings for sre_parse
    CATEGORY_CHARS = {
        sre_parse.CATEGORY_DIGIT: set('0123456789'),
        sre_parse.CATEGORY_NOT_DIGIT: "ANY",  # Matches too much to enumerate
        sre_parse.CATEGORY_SPACE: set(' \t\n\r\f\v'),
        sre_parse.CATEGORY_NOT_SPACE: "ANY",
        sre_parse.CATEGORY_WORD: set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'),
        sre_parse.CATEGORY_NOT_WORD: "ANY",
    }

    def first_chars(self, node: tuple) -> set[str] | Literal["ANY"]:
        """
        Compute set of possible first characters for a pattern node.

        Args:
            node: A tuple from sre_parse (op_code, value)

        Returns:
            Set of characters or "ANY" if it can match any character
        """
        if not node:
            return set()

        op = node[0]
        value = node[1] if len(node) > 1 else None

        if op == sre_parse.LITERAL:
            return {chr(value)}

        elif op == sre_parse.NOT_LITERAL:
            return "ANY"  # Matches everything except one char

        elif op == sre_parse.ANY:
            return "ANY"

        elif op == sre_parse.IN:
            # Character class like [abc] or [a-z]
            return self._expand_char_class(value)

        elif op == sre_parse.BRANCH:
            # Alternation - union of all branches
            result: set[str] = set()
            for branch in value[1]:
                if branch:
                    branch_chars = self.first_chars(branch[0])
                    if branch_chars == "ANY":
                        return "ANY"
                    result.update(branch_chars)
            return result

        elif op == sre_parse.SUBPATTERN:
            # Group - check first element
            _, _, _, items = value
            if items:
                return self.first_chars(items[0])
            return set()

        elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
            # Quantifier - check the quantified element
            _, _, items = value
            if items:
                return self.first_chars(items[0])
            return set()

        elif op == sre_parse.CATEGORY:
            return self.CATEGORY_CHARS.get(value, "ANY")

        elif op in (sre_parse.AT, sre_parse.ASSERT, sre_parse.ASSERT_NOT, sre_parse.AT_BEGINNING, sre_parse.AT_END):
            # Anchors and assertions don't match characters
            return set()

        elif op == sre_parse.GROUPREF:
            # Backreference - could match anything the group matched
            return "ANY"

        else:
            # Unknown op - be conservative
            return "ANY"

    def _expand_char_class(self, items: list) -> set[str] | Literal["ANY"]:
        """Expand a character class into a set of characters."""
        result: set[str] = set()
        negate = False

        for item in items:
            op = item[0]
            value = item[1] if len(item) > 1 else None

            if op == sre_parse.NEGATE:
                negate = True

            elif op == sre_parse.LITERAL:
                result.add(chr(value))

            elif op == sre_parse.RANGE:
                start, end = value
                for i in range(start, end + 1):
                    result.add(chr(i))
                    if len(result) > 100:  # Too many chars, treat as ANY
                        return "ANY"

            elif op == sre_parse.CATEGORY:
                cat_chars = self.CATEGORY_CHARS.get(value, "ANY")
                if cat_chars == "ANY":
                    return "ANY"
                result.update(cat_chars)

        if negate:
            return "ANY"  # Negated classes match too much

        return result

    def can_overlap(self, node1: tuple, node2: tuple) -> bool:
        """Check if two pattern elements can match the same character."""
        chars1 = self.first_chars(node1)
        chars2 = self.first_chars(node2)

        if chars1 == "ANY" or chars2 == "ANY":
            return True
        return bool(chars1 & chars2)


class VulnerabilityDetector:
    """
    Detects ReDoS vulnerabilities in parsed regex AST.

    Detection is based on the following vulnerability patterns from ReDoS research:

    1. Nested Quantifiers (NQ): O(2^n)
       Pattern: (a+)+ where quantifier contains another quantifier
       Why: Engine must try all ways to distribute matches between levels

    2. Exponential Overlapping Disjunction (EOD): O(2^n)
       Pattern: (a|ab)+ where alternation branches can match same prefix
       Why: At each position, engine can choose either branch

    3. Polynomial Overlapping Adjacency (POA): O(n^k)
       Pattern: .*.*  where k = number of adjacent greedy quantifiers
       Why: Engine tries all ways to divide input between quantifiers
    """

    def __init__(self):
        self.char_analyzer = CharacterSetAnalyzer()

    def detect_all(self, ast: sre_parse.SubPattern, pattern: str) -> list[RegexIssue]:
        """Run all vulnerability detectors on the AST."""
        issues = []
        issues.extend(self._detect_nested_quantifiers(ast, pattern))
        issues.extend(self._detect_overlapping_disjunction(ast, pattern))
        issues.extend(self._detect_greedy_chains(ast, pattern))
        issues.extend(self._detect_backreferences(ast, pattern))
        issues.extend(self._detect_lookarounds(ast, pattern))
        issues.extend(self._detect_large_quantifiers(ast, pattern))
        return issues

    def _detect_nested_quantifiers(self, ast: sre_parse.SubPattern, pattern: str) -> list[RegexIssue]:
        """
        Detect nested quantifier patterns (NQ) - O(2^n) complexity.

        A nested quantifier occurs when a quantified group contains another
        quantified element. Examples: (a+)+, (a*)+, ((a|b)+)*

        The regex engine must try all possible ways to distribute the input
        between the inner and outer quantifiers, leading to exponential paths.
        """
        issues = []

        def check_node(node: tuple, in_quantifier: bool = False, depth: int = 0) -> None:
            if not node or not isinstance(node, tuple):
                return

            op = node[0]
            value = node[1] if len(node) > 1 else None

            if op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                min_count, max_count, items = value

                # Check if this is a repeating quantifier (not {1} or {0,1})
                is_repeating = max_count is None or max_count > 1 or (max_count == sre_parse.MAXREPEAT)

                if is_repeating:
                    if in_quantifier:
                        # Found nested quantifier!
                        segment = self._extract_segment(items, pattern)
                        issues.append(
                            RegexIssue(
                                type=IssueType.NESTED_QUANTIFIER,
                                severity=Severity.CRITICAL,
                                complexity_class="exponential",
                                complexity_notation="O(2^n)",
                                segment=segment,
                                explanation=(
                                    "Nested quantifiers create exponential backtracking. "
                                    "The regex engine must try all possible ways to distribute "
                                    "matches between inner and outer repetitions. For an input "
                                    "of length n, this can result in 2^n matching attempts."
                                ),
                                fix_suggestions=[
                                    "Remove the nested quantifier by flattening the pattern",
                                    "Use atomic grouping (requires 'regex' module): (?>...)+",
                                    "Make the inner pattern non-overlapping with the outer",
                                    "Use possessive quantifiers if available: a++",
                                ],
                            )
                        )

                    # Recurse into the quantified content with in_quantifier=True
                    for item in items:
                        check_node(item, in_quantifier=True, depth=depth + 1)
                else:
                    # Fixed quantifier like {1} - recurse normally
                    for item in items:
                        check_node(item, in_quantifier=in_quantifier, depth=depth + 1)

            elif op == sre_parse.SUBPATTERN:
                _, _, _, items = value
                for item in items:
                    check_node(item, in_quantifier=in_quantifier, depth=depth + 1)

            elif op == sre_parse.BRANCH:
                for branch in value[1]:
                    for item in branch:
                        check_node(item, in_quantifier=in_quantifier, depth=depth + 1)

            elif op in (sre_parse.ASSERT, sre_parse.ASSERT_NOT):
                _, items = value
                for item in items:
                    check_node(item, in_quantifier=in_quantifier, depth=depth + 1)

        for node in ast:
            check_node(node)

        return issues

    def _detect_overlapping_disjunction(self, ast: sre_parse.SubPattern, pattern: str) -> list[RegexIssue]:
        """
        Detect exponential overlapping disjunction (EOD) - O(2^n) complexity.

        This occurs when alternation branches inside a quantifier can match
        the same characters. Examples: (a|a)+, (a|ab)+, (\\w|\\d)+

        At each position, the engine can choose either branch, and if both
        can match, all combinations must be tried on failure.

        Note: Python's sre_parse optimizes some patterns:
        - (a|b)+ becomes [ab]+ (character class)
        - (a|ab)+ becomes a(|b)+ (factored prefix with optional suffix)
        - (a|a)+ becomes a(|)+ (fully optimized)
        - (\\d|\\w)+ becomes [\\d\\w]+ (merged character class)

        We detect the factored form where a literal/pattern is followed by
        a BRANCH with an empty first alternative inside a quantifier.
        """
        issues = []

        def check_node(node: tuple, in_quantifier: bool = False) -> None:
            if not node or not isinstance(node, tuple):
                return

            op = node[0]
            value = node[1] if len(node) > 1 else None

            if op == sre_parse.BRANCH and in_quantifier:
                branches = value[1]
                if len(branches) >= 2:
                    # Check for factored form: (a|ab)+ parses as a followed by (|b)
                    # If one branch is empty, this indicates overlapping alternatives
                    empty_branches = [b for b in branches if len(b) == 0]
                    non_empty_branches = [b for b in branches if len(b) > 0]

                    # Case 1: All branches empty - this is (a|a)+ fully optimized to a(|)+
                    if len(empty_branches) >= 2 and len(non_empty_branches) == 0:
                        issues.append(
                            RegexIssue(
                                type=IssueType.OVERLAPPING_DISJUNCTION,
                                severity=Severity.CRITICAL,
                                complexity_class="exponential",
                                complexity_notation="O(2^n)",
                                segment="(identical alternation)",
                                explanation=(
                                    "Alternation with identical branches inside a quantifier. "
                                    "Pattern like (a|a)+ creates exponential backtracking because "
                                    "at each position the engine can choose either branch, and all "
                                    "combinations must be tried on failure."
                                ),
                                fix_suggestions=[
                                    "Remove duplicate alternatives: (a|a)+ -> a+",
                                    "This pattern appears to have redundant alternation",
                                ],
                            )
                        )
                        return

                    # Case 2: Mix of empty and non-empty - this is (a|ab)+ factored form
                    if len(empty_branches) > 0 and len(non_empty_branches) > 0:
                        # This is the factored form of overlapping alternation
                        # e.g., (a|ab)+ -> a(|b)+ where empty branch means "match nothing more"
                        issues.append(
                            RegexIssue(
                                type=IssueType.OVERLAPPING_DISJUNCTION,
                                severity=Severity.CRITICAL,
                                complexity_class="exponential",
                                complexity_notation="O(2^n)",
                                segment="(overlapping alternation)",
                                explanation=(
                                    "Alternation with optional suffix inside a quantifier causes "
                                    "exponential backtracking. The regex engine factored out the "
                                    "common prefix, but the pattern still has ambiguity: at each "
                                    "position it can match the short or long form. Pattern like "
                                    "(a|ab)+ or (x|xy)+ have this issue."
                                ),
                                fix_suggestions=[
                                    "Make alternatives mutually exclusive",
                                    "Use atomic grouping if available: (?>a|ab)+",
                                    "Rewrite to avoid optional suffixes in repetition",
                                    "Consider if the longer alternative is actually needed",
                                ],
                            )
                        )
                        return

                    # Case 3: Check for truly overlapping non-empty branches
                    for i, branch1 in enumerate(non_empty_branches):
                        for branch2 in non_empty_branches[i + 1 :]:
                            if self.char_analyzer.can_overlap(branch1[0], branch2[0]):
                                segment = self._format_branches(branches)
                                issues.append(
                                    RegexIssue(
                                        type=IssueType.OVERLAPPING_DISJUNCTION,
                                        severity=Severity.CRITICAL,
                                        complexity_class="exponential",
                                        complexity_notation="O(2^n)",
                                        segment=segment,
                                        explanation=(
                                            "Alternation branches can match the same characters, "
                                            "causing exponential backtracking when inside a quantifier. "
                                            "The engine must try all combinations of which branch "
                                            "matched at each position."
                                        ),
                                        fix_suggestions=[
                                            "Make branches mutually exclusive (different starting chars)",
                                            "Reorder branches so the most specific comes first",
                                            "Factor out common prefixes: (ab|ac) -> a(b|c)",
                                            "Use atomic grouping if available: (?>a|ab)+",
                                        ],
                                    )
                                )
                                return  # One issue per branch node is enough

            elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                min_count, max_count, items = value
                is_repeating = max_count is None or max_count > 1
                for item in items:
                    check_node(item, in_quantifier=is_repeating)

            elif op == sre_parse.SUBPATTERN:
                _, _, _, items = value
                for item in items:
                    check_node(item, in_quantifier=in_quantifier)

            elif op == sre_parse.BRANCH:
                for branch in value[1]:
                    for item in branch:
                        check_node(item, in_quantifier=in_quantifier)

        for node in ast:
            check_node(node)

        return issues

    def _detect_greedy_chains(self, ast: sre_parse.SubPattern, pattern: str) -> list[RegexIssue]:
        """
        Detect polynomial overlapping adjacency (POA) - O(n^k) complexity.

        This occurs when multiple greedy quantifiers appear in sequence and
        can match overlapping content. Examples: .*.*  (O(n²)), .*.*.* (O(n³))

        The engine must try all ways to divide the input between quantifiers.
        """
        issues = []

        # Find sequences of greedy quantifiers that can match any character
        greedy_any_count = 0
        any_positions = []

        def is_greedy_any(node: tuple) -> bool:
            """Check if node is a greedy quantifier matching any character."""
            if not node or not isinstance(node, tuple):
                return False
            op = node[0]
            if op == sre_parse.MAX_REPEAT:  # Greedy (not MIN_REPEAT which is lazy)
                min_count, max_count, items = node[1]
                if max_count is None or max_count > 1:
                    if items and items[0][0] == sre_parse.ANY:
                        return True
                    # Also check for broad character classes
                    if items and items[0][0] == sre_parse.IN:
                        chars = self.char_analyzer.first_chars(items[0])
                        if chars == "ANY":
                            return True
            return False

        def count_greedy_sequence(items: list) -> int:
            """Count consecutive greedy-any quantifiers."""
            count = 0
            for node in items:
                if is_greedy_any(node):
                    count += 1
                elif node[0] == sre_parse.SUBPATTERN:
                    # Look inside groups
                    _, _, _, sub_items = node[1]
                    count += count_greedy_sequence(sub_items)
            return count

        # Check top level and inside groups
        def analyze_node(node: tuple) -> None:
            nonlocal greedy_any_count
            if not node or not isinstance(node, tuple):
                return

            op = node[0]

            if is_greedy_any(node):
                greedy_any_count += 1

            elif op == sre_parse.SUBPATTERN:
                _, _, _, items = node[1]
                for item in items:
                    analyze_node(item)

            elif op == sre_parse.BRANCH:
                for branch in node[1][1]:
                    branch_count = 0
                    for item in branch:
                        if is_greedy_any(item):
                            branch_count += 1
                        analyze_node(item)

        for node in ast:
            analyze_node(node)

        if greedy_any_count >= 2:
            degree = greedy_any_count
            notation = f"O(n^{degree})" if degree <= 3 else f"O(n^{degree})"

            # Determine severity based on degree
            if degree >= 4:
                severity = Severity.CRITICAL
            elif degree >= 3:
                severity = Severity.HIGH
            else:
                severity = Severity.MEDIUM

            issues.append(
                RegexIssue(
                    type=IssueType.GREEDY_CHAIN,
                    severity=severity,
                    complexity_class="polynomial",
                    complexity_notation=notation,
                    segment=f"{greedy_any_count}x greedy quantifiers (.*)",
                    explanation=(
                        f"Found {greedy_any_count} greedy quantifiers that can match any character. "
                        f"This creates {notation} polynomial complexity. For a 1000-character input, "
                        f"this could require up to {1000**degree:,} operations when the pattern fails to match."
                    ),
                    fix_suggestions=[
                        "Use lazy quantifiers (.*?) instead of greedy (.*)",
                        "Use specific character classes instead of '.' (e.g., [^\\n]* for non-newlines)",
                        "Add anchors (^ and $) to constrain matching",
                        "Break into multiple simpler patterns and search separately",
                    ],
                )
            )

        return issues

    def _detect_backreferences(self, ast: sre_parse.SubPattern, pattern: str) -> list[RegexIssue]:
        """
        Detect backreferences - technically NP-complete but usually fast.

        Backreferences like \\1 make the regex non-regular (not a true
        regular expression). While Python handles simple cases efficiently,
        complex backreference patterns can be slow.
        """
        issues = []
        backref_count = 0

        def check_node(node: tuple) -> None:
            nonlocal backref_count
            if not node or not isinstance(node, tuple):
                return

            op = node[0]
            value = node[1] if len(node) > 1 else None

            if op == sre_parse.GROUPREF:
                backref_count += 1

            elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                for item in value[2]:
                    check_node(item)

            elif op == sre_parse.SUBPATTERN:
                for item in value[3]:
                    check_node(item)

            elif op == sre_parse.BRANCH:
                for branch in value[1]:
                    for item in branch:
                        check_node(item)

        for node in ast:
            check_node(node)

        if backref_count > 0:
            issues.append(
                RegexIssue(
                    type=IssueType.BACKREFERENCE,
                    severity=Severity.MEDIUM,
                    complexity_class="linear",  # Usually, but can be worse
                    complexity_notation="O(n) typical",
                    segment=f"{backref_count} backreference(s)",
                    explanation=(
                        f"Found {backref_count} backreference(s). Backreferences make the pattern "
                        "non-regular (technically NP-complete). Python's regex engine handles simple "
                        "cases efficiently, but complex backreference patterns can be slow."
                    ),
                    fix_suggestions=[
                        "Consider if the backreference is necessary",
                        "Use a simpler pattern if the backreference is for validation only",
                        "For complex cases, consider a two-pass approach with simpler regexes",
                    ],
                )
            )

        return issues

    def _detect_lookarounds(self, ast: sre_parse.SubPattern, pattern: str) -> list[RegexIssue]:
        """
        Detect lookahead/lookbehind assertions.

        Lookarounds require the engine to re-scan portions of the input,
        which adds constant-factor overhead. Nested lookarounds multiply
        this overhead.
        """
        issues = []
        lookaround_count = 0
        nested_lookarounds = 0

        def check_node(node: tuple, in_lookaround: bool = False) -> None:
            nonlocal lookaround_count, nested_lookarounds
            if not node or not isinstance(node, tuple):
                return

            op = node[0]
            value = node[1] if len(node) > 1 else None

            if op in (sre_parse.ASSERT, sre_parse.ASSERT_NOT):
                lookaround_count += 1
                if in_lookaround:
                    nested_lookarounds += 1
                # Check contents for nested lookarounds
                _, items = value
                for item in items:
                    check_node(item, in_lookaround=True)

            elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                for item in value[2]:
                    check_node(item, in_lookaround=in_lookaround)

            elif op == sre_parse.SUBPATTERN:
                for item in value[3]:
                    check_node(item, in_lookaround=in_lookaround)

            elif op == sre_parse.BRANCH:
                for branch in value[1]:
                    for item in branch:
                        check_node(item, in_lookaround=in_lookaround)

        for node in ast:
            check_node(node)

        if nested_lookarounds > 0:
            issues.append(
                RegexIssue(
                    type=IssueType.NESTED_LOOKAROUND,
                    severity=Severity.MEDIUM,
                    complexity_class="linear",
                    complexity_notation="O(n) with high constant",
                    segment=f"{nested_lookarounds} nested lookaround(s)",
                    explanation=(
                        f"Found {nested_lookarounds} nested lookaround(s). Each lookaround level "
                        "requires re-scanning the input, multiplying the work done. While still "
                        "linear, the constant factor can be significant."
                    ),
                    fix_suggestions=[
                        "Flatten nested lookarounds if possible",
                        "Consider if the nested structure is necessary",
                        "Use anchored patterns to reduce scanning",
                    ],
                )
            )
        elif lookaround_count >= 3:
            issues.append(
                RegexIssue(
                    type=IssueType.MULTIPLE_LOOKAROUND,
                    severity=Severity.LOW,
                    complexity_class="linear",
                    complexity_notation="O(n)",
                    segment=f"{lookaround_count} lookaround(s)",
                    explanation=(
                        f"Found {lookaround_count} lookaround assertions. Each adds constant-factor "
                        "overhead as the engine must re-scan portions of the input. Consider if "
                        "all are necessary."
                    ),
                    fix_suggestions=[
                        "Combine lookaheads where possible",
                        "Consider if some lookarounds can be replaced with simpler patterns",
                    ],
                )
            )

        return issues

    def _detect_large_quantifiers(self, ast: sre_parse.SubPattern, pattern: str) -> list[RegexIssue]:
        """
        Detect extremely large or unbounded quantifiers.

        Quantifiers like {1000,} or {0,10000} can cause performance issues
        even without backtracking, simply due to the number of iterations.
        """
        issues = []

        def check_node(node: tuple) -> None:
            if not node or not isinstance(node, tuple):
                return

            op = node[0]
            value = node[1] if len(node) > 1 else None

            if op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                min_count, max_count, items = value

                # Check for very large explicit bounds
                # Note: MAXREPEAT is a special constant for unbounded quantifiers (* and +)
                # We only flag truly large explicit bounds like {0,10000}
                is_explicit_large = (
                    isinstance(max_count, int) and max_count >= 10000 and max_count != sre_parse.MAXREPEAT
                )
                if is_explicit_large:
                    issues.append(
                        RegexIssue(
                            type=IssueType.LARGE_QUANTIFIER,
                            severity=Severity.LOW,
                            complexity_class="linear",
                            complexity_notation="O(n)",
                            segment=f"{{,{max_count}}}",
                            explanation=(
                                f"Quantifier with large upper bound ({max_count}). While not "
                                "causing backtracking issues, very large quantifiers can be slow "
                                "due to the number of iterations required."
                            ),
                            fix_suggestions=[
                                "Consider if such a large bound is necessary",
                                "Use unbounded * or + if exact count doesn't matter",
                            ],
                        )
                    )

                for item in items:
                    check_node(item)

            elif op == sre_parse.SUBPATTERN:
                for item in value[3]:
                    check_node(item)

            elif op == sre_parse.BRANCH:
                for branch in value[1]:
                    for item in branch:
                        check_node(item)

        for node in ast:
            check_node(node)

        return issues

    def _extract_segment(self, items: list, pattern: str) -> str:
        """Try to extract the relevant pattern segment (best effort)."""
        if not items:
            return ""
        # This is a simplification - proper extraction would need position tracking
        return "(nested pattern)"

    def _format_branches(self, branches: list) -> str:
        """Format branches for display."""
        if len(branches) <= 3:
            return f"({len(branches)} branches)"
        return f"({len(branches)} branches)"


def calculate_star_height(ast: sre_parse.SubPattern) -> int:
    """
    Calculate the star height (quantifier nesting depth) of a regex.

    Star height is the maximum depth of nested quantifiers:
    - a+ has star height 1
    - (a+)+ has star height 2
    - ((a+)+)+ has star height 3

    Higher star height correlates with potential for exponential behavior.
    """
    max_height = 0

    def check_node(node: tuple, current_height: int) -> None:
        nonlocal max_height
        if not node or not isinstance(node, tuple):
            return

        op = node[0]
        value = node[1] if len(node) > 1 else None

        if op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
            min_count, max_count, items = value
            is_repeating = max_count is None or max_count > 1

            if is_repeating:
                new_height = current_height + 1
                max_height = max(max_height, new_height)
                for item in items:
                    check_node(item, new_height)
            else:
                for item in items:
                    check_node(item, current_height)

        elif op == sre_parse.SUBPATTERN:
            for item in value[3]:
                check_node(item, current_height)

        elif op == sre_parse.BRANCH:
            for branch in value[1]:
                for item in branch:
                    check_node(item, current_height)

        elif op in (sre_parse.ASSERT, sre_parse.ASSERT_NOT):
            for item in value[1]:
                check_node(item, current_height)

    for node in ast:
        check_node(node, 0)

    return max_height


def count_quantifiers(ast: sre_parse.SubPattern) -> int:
    """Count total number of quantifiers in the pattern."""
    count = 0

    def check_node(node: tuple) -> None:
        nonlocal count
        if not node or not isinstance(node, tuple):
            return

        op = node[0]
        value = node[1] if len(node) > 1 else None

        if op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
            count += 1
            for item in value[2]:
                check_node(item)

        elif op == sre_parse.SUBPATTERN:
            for item in value[3]:
                check_node(item)

        elif op == sre_parse.BRANCH:
            for branch in value[1]:
                for item in branch:
                    check_node(item)

    for node in ast:
        check_node(node)

    return count


def has_anchors(ast: sre_parse.SubPattern) -> tuple[bool, bool]:
    """Check if pattern has start (^) and end ($) anchors."""
    has_start = False
    has_end = False

    if ast:
        first = ast[0]
        if first[0] == sre_parse.AT:
            if first[1] in (sre_parse.AT_BEGINNING, sre_parse.AT_BEGINNING_STRING):
                has_start = True

        last = ast[-1]
        if last[0] == sre_parse.AT:
            if last[1] in (sre_parse.AT_END, sre_parse.AT_END_STRING):
                has_end = True

    return has_start, has_end


def calculate_regex_complexity(regex: str) -> dict:
    """
    Analyze a regex pattern for complexity and ReDoS vulnerabilities.

    This function uses AST-based analysis via Python's sre_parse module to
    accurately detect patterns that can cause catastrophic backtracking or
    poor performance. The analysis is based on academic research into ReDoS
    vulnerabilities and regex engine behavior.

    Complexity Classes:

    1. EXPONENTIAL O(2^n) - CRITICAL:
       - Nested quantifiers: (a+)+, (a*)*
       - Overlapping disjunction: (a|ab)+
       These patterns can hang on inputs as short as 25-30 characters.

    2. POLYNOMIAL O(n^k) - HIGH to MEDIUM:
       - Greedy chains: .*.* is O(n²), .*.*.* is O(n³)
       For k>=3, becomes impractical on large inputs.

    3. LINEAR O(n) - LOW:
       - Most patterns without the above issues
       - Lookarounds add constant-factor overhead
       - Backreferences are technically NP-complete but usually fast

    Args:
        regex: The regular expression pattern to analyze

    Returns:
        Dictionary containing:
        - score: Numeric complexity score (0-100 scale)
        - risk_level: "safe", "caution", "warning", "dangerous", "critical"
        - complexity_class: "linear", "polynomial", "exponential"
        - complexity_notation: Big-O notation string
        - issues: List of detected issues with explanations
        - recommendations: Actionable fix suggestions
        - performance: Estimated operations for different input sizes
        - star_height: Maximum quantifier nesting depth
        - pattern_length: Length of the pattern
        - has_anchors: Whether pattern has ^ and $ anchors

    Example:
        >>> result = calculate_regex_complexity("(a+)+")
        >>> result['risk_level']
        'critical'
        >>> result['complexity_class']
        'exponential'
        >>> result['issues'][0]['explanation']
        'Nested quantifiers create exponential backtracking...'

    References:
        - https://github.com/doyensec/regexploit
        - https://www.regular-expressions.info/catastrophic.html
        - https://swtch.com/~rsc/regexp/regexp1.html
    """
    # Handle invalid regex
    try:
        ast = sre_parse.parse(regex)
    except Exception as e:
        return {
            'score': 0,
            'risk_level': 'error',
            'complexity_class': 'unknown',
            'complexity_notation': 'N/A',
            'issues': [
                {
                    'type': 'parse_error',
                    'severity': 'error',
                    'explanation': f"Invalid regex pattern: {e}",
                    'fix_suggestions': ["Check regex syntax"],
                }
            ],
            'recommendations': [f"Fix syntax error: {e}"],
            'performance': {
                'ops_at_100': 0,
                'ops_at_1000': 0,
                'ops_at_10000': 0,
                'safe_for_large_files': False,
            },
            'star_height': 0,
            'pattern_length': len(regex),
            'has_anchors': (False, False),
            # Legacy fields
            'level': 'error',
            'risk': f'Invalid pattern: {e}',
            'warnings': [f'Parse error: {e}'],
            'details': {},
        }

    # Run vulnerability detection
    detector = VulnerabilityDetector()
    issues = detector.detect_all(ast, regex)

    # Calculate metrics
    star_height = calculate_star_height(ast)
    quantifier_count = count_quantifiers(ast)
    anchors = has_anchors(ast)

    # Determine overall complexity class and score
    has_exponential = any(i.complexity_class == "exponential" for i in issues)
    has_polynomial = any(i.complexity_class == "polynomial" for i in issues)

    polynomial_degree = 1
    for issue in issues:
        if issue.type == IssueType.GREEDY_CHAIN:
            # Extract degree from notation like "O(n^2)"
            notation = issue.complexity_notation
            if "^" in notation:
                try:
                    degree = int(notation.split("^")[1].rstrip(")"))
                    polynomial_degree = max(polynomial_degree, degree)
                except (ValueError, IndexError):
                    polynomial_degree = max(polynomial_degree, 2)

    # Calculate score (0-100 scale)
    score = 0

    if has_exponential:
        complexity_class = "exponential"
        complexity_notation = "O(2^n)"
        score = 90  # Base score for exponential
        # Add points for multiple exponential issues
        exp_count = sum(1 for i in issues if i.complexity_class == "exponential")
        score = min(100, score + (exp_count - 1) * 5)

    elif has_polynomial:
        complexity_class = "polynomial"
        complexity_notation = f"O(n^{polynomial_degree})"
        # Score based on polynomial degree
        if polynomial_degree >= 4:
            score = 80
        elif polynomial_degree == 3:
            score = 65
        else:  # degree 2
            score = 50
    else:
        complexity_class = "linear"
        complexity_notation = "O(n)"
        # Base score for linear patterns
        score = 10
        # Add small amounts for various features
        score += min(20, len(issues) * 5)  # Issues add some complexity
        score += min(10, star_height * 3)  # Star height adds some
        score += min(10, quantifier_count)  # Many quantifiers add some

    # Adjust for anchors (they help constrain matching)
    if anchors[0] and anchors[1]:
        score = max(0, score - 5)

    # Determine risk level
    if score >= 85:
        risk_level = "critical"
        risk_description = "CRITICAL - ReDoS vulnerability, exponential backtracking"
    elif score >= 70:
        risk_level = "dangerous"
        risk_description = "Dangerous - severe performance impact likely"
    elif score >= 50:
        risk_level = "warning"
        risk_description = "Warning - polynomial complexity, may be slow on large inputs"
    elif score >= 25:
        risk_level = "caution"
        risk_description = "Caution - some complexity, monitor on large files"
    else:
        risk_level = "safe"
        risk_description = "Safe - linear complexity, suitable for large files"

    # Build recommendations list
    recommendations = []
    seen_suggestions = set()
    for issue in issues:
        for suggestion in issue.fix_suggestions:
            if suggestion not in seen_suggestions:
                recommendations.append(suggestion)
                seen_suggestions.add(suggestion)

    if not recommendations:
        if risk_level == "safe":
            recommendations = ["Pattern looks safe for production use"]
        else:
            recommendations = ["Review pattern for potential optimizations"]

    # Build performance estimates
    if complexity_class == "exponential":
        perf = PerformanceEstimate.from_complexity("exponential")
    elif complexity_class == "polynomial":
        perf = PerformanceEstimate.from_complexity("polynomial", polynomial_degree)
    else:
        perf = PerformanceEstimate.from_complexity("linear")

    # Format issues for output
    formatted_issues = []
    warnings = []  # Legacy format
    for issue in issues:
        formatted_issues.append(
            {
                'type': issue.type.value,
                'severity': issue.severity.value,
                'complexity_class': issue.complexity_class,
                'complexity_notation': issue.complexity_notation,
                'segment': issue.segment,
                'explanation': issue.explanation,
                'fix_suggestions': issue.fix_suggestions,
            }
        )
        # Legacy warning format
        severity_prefix = {
            Severity.CRITICAL: "CRITICAL",
            Severity.HIGH: "HIGH",
            Severity.MEDIUM: "MEDIUM",
            Severity.LOW: "LOW",
        }
        warnings.append(f"[{severity_prefix[issue.severity]}] {issue.type.value}: {issue.explanation[:100]}...")

    # Build legacy details dict
    details = {
        'star_height': star_height,
        'quantifier_count': quantifier_count,
        'has_start_anchor': anchors[0],
        'has_end_anchor': anchors[1],
        'issue_count': len(issues),
    }

    # Map risk_level to legacy level
    level_map = {
        'safe': 'very_simple',
        'caution': 'simple',
        'warning': 'moderate',
        'dangerous': 'complex',
        'critical': 'dangerous',
    }

    return {
        # New fields
        'score': round(score, 1),
        'risk_level': risk_level,
        'complexity_class': complexity_class,
        'complexity_notation': complexity_notation,
        'issues': formatted_issues,
        'recommendations': recommendations,
        'performance': {
            'ops_at_100': perf.ops_at_100,
            'ops_at_1000': perf.ops_at_1000,
            'ops_at_10000': perf.ops_at_10000,
            'safe_for_large_files': perf.safe_for_large_files,
        },
        'star_height': star_height,
        'pattern_length': len(regex),
        'has_anchors': anchors,
        # Legacy fields for compatibility
        'level': level_map.get(risk_level, 'moderate'),
        'risk': risk_description,
        'warnings': warnings if warnings else [],
        'details': details,
    }
