"""
Template Rendering System.

This module provides template-based code generation using Jinja2 templates.
It includes:
- TemplateRenderer: Main template rendering engine
- Template validation and error handling
- Choreo-specific templates for different DSL components

Templates are organized by target language/dialect:
- choreo/: Choreo DSL templates
- cuda/: CUDA templates (future)
- opencl/: OpenCL templates (future)
"""

from .renderer import (
    TemplateRenderer,
    JinjaTemplateRenderer,
    SimpleTemplateRenderer,
    create_template_renderer,
    validate_template_syntax,
)

__all__ = [
    "TemplateRenderer",
    "JinjaTemplateRenderer",
    "SimpleTemplateRenderer",
    "create_template_renderer",
    "validate_template_syntax",
]
