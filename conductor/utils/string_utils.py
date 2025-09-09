"""
String Manipulation Utilities for Conductor Framework.

This module provides general-purpose string manipulation functions used
throughout the project for formatting, parsing, and text processing.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Union
from .constants import TEMPLATE_INDENT


# =============================================================================
# Text Formatting and Indentation
# =============================================================================

def indent_text(text: str, level: int = 1, indent_str: str = TEMPLATE_INDENT) -> str:
    """
    Indent text by the specified level.
    
    Args:
        text: Text to indent
        level: Indentation level (number of indent_str to prepend)
        indent_str: String to use for each indentation level
    
    Returns:
        Indented text
    """
    if not text:
        return text
    
    indent = indent_str * level
    lines = text.split('\n')
    indented_lines = [f"{indent}{line}" if line.strip() else line for line in lines]
    return '\n'.join(indented_lines)


def dedent_text(text: str) -> str:
    """
    Remove common leading whitespace from all lines.
    
    Args:
        text: Text to dedent
    
    Returns:
        Dedented text
    """
    import textwrap
    return textwrap.dedent(text)


def format_multiline_string(text: str, width: int = 80, indent_level: int = 0) -> str:
    """
    Format a multiline string with proper wrapping and indentation.
    
    Args:
        text: Text to format
        width: Maximum line width
        indent_level: Indentation level
    
    Returns:
        Formatted text
    """
    import textwrap
    
    # Dedent first to normalize
    text = dedent_text(text)
    
    # Wrap lines
    wrapped = textwrap.fill(text, width=width)
    
    # Apply indentation if needed
    if indent_level > 0:
        wrapped = indent_text(wrapped, indent_level)
    
    return wrapped


# =============================================================================
# Template and Code Generation Utilities
# =============================================================================

def format_code_block(code: str, indent_level: int = 1) -> str:
    """
    Format a code block with proper indentation.
    
    Args:
        code: Code to format
        indent_level: Indentation level
    
    Returns:
        Formatted code block
    """
    if not code.strip():
        return ""
    
    # Normalize the code first
    code = dedent_text(code).strip()
    
    # Apply indentation
    return indent_text(code, indent_level)


def join_code_lines(lines: List[str], separator: str = "\n") -> str:
    """
    Join code lines, filtering out empty lines.
    
    Args:
        lines: List of code lines
        separator: Separator to use between lines
    
    Returns:
        Joined code string
    """
    # Filter out None and empty lines
    filtered_lines = [line for line in lines if line is not None and line.strip()]
    return separator.join(filtered_lines)


def format_template_substitution(template: str, **kwargs) -> str:
    """
    Perform template substitution with error handling.
    
    Args:
        template: Template string with {key} placeholders
        **kwargs: Values to substitute
    
    Returns:
        Formatted string
    
    Raises:
        ValueError: If template substitution fails
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template parameter: {e}")
    except Exception as e:
        raise ValueError(f"Template substitution failed: {e}")


# =============================================================================
# Parsing and Extraction Utilities
# =============================================================================

def extract_function_name(code: str) -> Optional[str]:
    """
    Extract function name from code.
    
    Args:
        code: Code string
    
    Returns:
        Function name if found, None otherwise
    """
    # Pattern for C/C++ style functions
    cpp_pattern = r'(?:__co__\s+)?(?:\w+\s+)?(\w+)\s*\('
    match = re.search(cpp_pattern, code)
    if match:
        return match.group(1)
    
    # Pattern for Python functions
    python_pattern = r'def\s+(\w+)\s*\('
    match = re.search(python_pattern, code)
    if match:
        return match.group(1)
    
    return None


def extract_includes(code: str) -> List[str]:
    """
    Extract #include statements from code.
    
    Args:
        code: Code string
    
    Returns:
        List of include statements
    """
    pattern = r'#include\s*[<"]([^>"]+)[>"]'
    matches = re.findall(pattern, code)
    return matches


def parse_tensor_shape(shape_str: str) -> List[int]:
    """
    Parse tensor shape string into list of dimensions.
    
    Args:
        shape_str: Shape string like "[16, 32]" or "(16, 32)"
    
    Returns:
        List of dimension sizes
    
    Raises:
        ValueError: If shape string is invalid
    """
    # Remove brackets/parentheses and split
    cleaned = re.sub(r'[\[\]()]', '', shape_str)
    parts = [part.strip() for part in cleaned.split(',') if part.strip()]
    
    try:
        return [int(part) for part in parts]
    except ValueError as e:
        raise ValueError(f"Invalid shape string '{shape_str}': {e}")


# =============================================================================
# Validation and Sanitization
# =============================================================================

def is_valid_identifier(name: str) -> bool:
    """
    Check if a string is a valid identifier.
    
    Args:
        name: String to check
    
    Returns:
        True if valid identifier, False otherwise
    """
    if not name:
        return False
    
    # Must start with letter or underscore
    if not (name[0].isalpha() or name[0] == '_'):
        return False
    
    # Rest must be alphanumeric or underscore
    return all(c.isalnum() or c == '_' for c in name[1:])


def sanitize_for_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use as a filename.
    
    Args:
        name: String to sanitize
    
    Returns:
        Sanitized filename
    """
    # Replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with optional suffix.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    if len(suffix) >= max_length:
        return text[:max_length]
    
    return text[:max_length - len(suffix)] + suffix


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_size_bytes(size_bytes: int) -> str:
    """
    Format byte size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_duration_ms(duration_ms: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        duration_ms: Duration in milliseconds
    
    Returns:
        Formatted duration string
    """
    if duration_ms < 1:
        return f"{duration_ms * 1000:.1f}μs"
    elif duration_ms < 1000:
        return f"{duration_ms:.1f}ms"
    else:
        return f"{duration_ms / 1000:.1f}s"


def format_list_items(items: List[Any], max_items: int = 5, separator: str = ", ") -> str:
    """
    Format a list of items with optional truncation.
    
    Args:
        items: List of items to format
        max_items: Maximum number of items to show
        separator: Separator between items
    
    Returns:
        Formatted string
    """
    if not items:
        return "[]"
    
    str_items = [str(item) for item in items]
    
    if len(str_items) <= max_items:
        return f"[{separator.join(str_items)}]"
    else:
        shown = str_items[:max_items]
        remaining = len(str_items) - max_items
        return f"[{separator.join(shown)}, ... ({remaining} more)]"


# =============================================================================
# Dictionary and Data Structure Formatting
# =============================================================================

def format_dict_compact(data: Dict[str, Any], max_items: int = 5) -> str:
    """
    Format dictionary in compact form.
    
    Args:
        data: Dictionary to format
        max_items: Maximum number of items to show
    
    Returns:
        Formatted string
    """
    if not data:
        return "{}"
    
    items = list(data.items())
    if len(items) <= max_items:
        formatted_items = [f"{k}={v}" for k, v in items]
        return f"{{{', '.join(formatted_items)}}}"
    else:
        shown_items = [f"{k}={v}" for k, v in items[:max_items]]
        remaining = len(items) - max_items
        return f"{{{', '.join(shown_items)}, ... ({remaining} more)}}"


def format_nested_dict(data: Dict[str, Any], indent_level: int = 0) -> str:
    """
    Format nested dictionary with proper indentation.
    
    Args:
        data: Dictionary to format
        indent_level: Current indentation level
    
    Returns:
        Formatted string
    """
    if not data:
        return "{}"
    
    lines = ["{"]
    for key, value in data.items():
        if isinstance(value, dict):
            nested = format_nested_dict(value, indent_level + 1)
            lines.append(f"{TEMPLATE_INDENT * (indent_level + 1)}{key}: {nested}")
        else:
            lines.append(f"{TEMPLATE_INDENT * (indent_level + 1)}{key}: {value}")
    lines.append(f"{TEMPLATE_INDENT * indent_level}}}")
    
    return "\n".join(lines)
