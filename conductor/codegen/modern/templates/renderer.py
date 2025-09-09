"""
Template Rendering Engine.

This module provides template-based code generation using Jinja2 templates.
It includes validation, error handling, and support for custom filters and
functions specific to DSL generation.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
from pathlib import Path
import os

try:
    from jinja2 import Environment, FileSystemLoader, Template, TemplateError
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from ..types import TemplateRenderer, ValidationResult


class JinjaTemplateRenderer:
    """Jinja2-based template renderer for DSL generation."""
    
    def __init__(self, template_dir: Optional[str] = None):
        """Initialize the template renderer."""
        if not JINJA2_AVAILABLE:
            raise ImportError("Jinja2 is required for template rendering. Install with: pip install jinja2")
        
        # Set up template directory
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), "choreo")
        
        self._template_dir = Path(template_dir)
        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        
        # Add custom filters and functions
        self._setup_custom_filters()
    
    def _setup_custom_filters(self) -> None:
        """Set up custom Jinja2 filters for DSL generation."""
        
        def indent_filter(text: str, width: int = 2, first: bool = False) -> str:
            """Indent text by the specified width."""
            lines = text.splitlines()
            if not lines:
                return text
            
            indent = " " * width
            if first:
                return "\n".join(indent + line for line in lines)
            else:
                return lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])
        
        def shape_filter(shape: tuple) -> str:
            """Format a shape tuple for DSL output."""
            return f"[{', '.join(map(str, shape))}]"
        
        def dtype_filter(dtype) -> str:
            """Convert PyTorch dtype to DSL type string."""
            import torch
            if dtype == torch.float32:
                return "f32"
            elif dtype == torch.float64:
                return "f64"
            elif dtype == torch.int32:
                return "i32"
            elif dtype == torch.int64:
                return "i64"
            else:
                return "f32"  # Default fallback
        
        def join_with_commas(items: List[str]) -> str:
            """Join items with commas and proper spacing."""
            return ", ".join(items)
        
        # Register filters
        self._env.filters["indent"] = indent_filter
        self._env.filters["shape"] = shape_filter
        self._env.filters["dtype"] = dtype_filter
        self._env.filters["join_commas"] = join_with_commas
        
        # Add global functions
        self._env.globals["range"] = range
        self._env.globals["len"] = len
        self._env.globals["enumerate"] = enumerate
    
    def render(self, template: str, context: Dict[str, Any]) -> str:
        """Render a template string with the given context."""
        try:
            template_obj = Template(template, environment=self._env)
            return template_obj.render(**context)
        except TemplateError as e:
            raise ValueError(f"Template rendering failed: {e}")
    
    def render_file(self, template_path: str, context: Dict[str, Any]) -> str:
        """Render a template file with the given context."""
        try:
            template = self._env.get_template(template_path)
            return template.render(**context)
        except TemplateError as e:
            raise ValueError(f"Template file rendering failed: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load template file '{template_path}': {e}")
    
    def validate_template(self, template: str) -> ValidationResult:
        """Validate a template string."""
        try:
            Template(template, environment=self._env)
            return ValidationResult(
                is_valid=True,
                errors=(),
                warnings=(),
                metadata={"template_length": len(template)},
            )
        except TemplateError as e:
            return ValidationResult(
                is_valid=False,
                errors=(str(e),),
                warnings=(),
                metadata={"template_length": len(template)},
            )
    
    def list_templates(self) -> List[str]:
        """List available template files."""
        try:
            return self._env.list_templates()
        except Exception:
            return []


class SimpleTemplateRenderer:
    """Simple string-based template renderer for basic use cases."""
    
    def render(self, template: str, context: Dict[str, Any]) -> str:
        """Render a template using simple string formatting."""
        try:
            return template.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}")
    
    def render_file(self, template_path: str, context: Dict[str, Any]) -> str:
        """Render a template file using simple string formatting."""
        try:
            with open(template_path, 'r') as f:
                template = f.read()
            return self.render(template, context)
        except FileNotFoundError:
            raise ValueError(f"Template file not found: {template_path}")
        except Exception as e:
            raise ValueError(f"Failed to render template file: {e}")


def create_template_renderer(use_jinja: bool = True, template_dir: Optional[str] = None) -> TemplateRenderer:
    """Create a template renderer."""
    if use_jinja and JINJA2_AVAILABLE:
        return JinjaTemplateRenderer(template_dir)
    else:
        return SimpleTemplateRenderer()


def validate_template_syntax(template: str, renderer: Optional[TemplateRenderer] = None) -> ValidationResult:
    """Validate template syntax."""
    if renderer is None:
        renderer = create_template_renderer()
    
    if hasattr(renderer, 'validate_template'):
        return renderer.validate_template(template)
    else:
        # Basic validation for simple renderer
        try:
            renderer.render(template, {})
            return ValidationResult(
                is_valid=True,
                errors=(),
                warnings=(),
                metadata={},
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=(str(e),),
                warnings=(),
                metadata={},
            )
