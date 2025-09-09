"""
Base Classes for Code Generators.

This module provides base classes and utilities for implementing
specialized code generators in the modern DSL architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from ..types import (
    CodeGenerator,
    CodeGenerationContext,
    CodeFragment,
    TemplateRenderer,
)


class BaseCodeGenerator(ABC):
    """Base class for all code generators."""
    
    def __init__(self, template_renderer: TemplateRenderer):
        """Initialize the generator with a template renderer."""
        self._template_renderer = template_renderer
    
    @abstractmethod
    def generate(self, context: CodeGenerationContext) -> CodeFragment:
        """Generate a code fragment for the given context."""
        pass
    
    def _render_template(self, template: str, template_context: Dict[str, Any]) -> str:
        """Render a template with the given context."""
        return self._template_renderer.render(template, template_context)
    
    def _render_template_file(self, template_path: str, template_context: Dict[str, Any]) -> str:
        """Render a template file with the given context."""
        return self._template_renderer.render_file(template_path, template_context)
    
    def _create_fragment(
        self,
        content: str,
        dependencies: Optional[List[str]] = None,
        declarations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CodeFragment:
        """Create a code fragment with the given content and metadata."""
        return CodeFragment(
            content=content,
            dependencies=tuple(dependencies or []),
            declarations=tuple(declarations or []),
            metadata=metadata or {},
        )


class TemplateBasedGenerator(BaseCodeGenerator):
    """Base class for template-based generators."""
    
    def __init__(self, template_renderer: TemplateRenderer, template_name: str):
        """Initialize with a template renderer and template name."""
        super().__init__(template_renderer)
        self._template_name = template_name
    
    def generate(self, context: CodeGenerationContext) -> CodeFragment:
        """Generate code using the configured template."""
        template_context = self._build_template_context(context)
        content = self._render_template_file(self._template_name, template_context)
        
        return self._create_fragment(
            content=content,
            dependencies=self._get_dependencies(context),
            declarations=self._get_declarations(context),
            metadata=self._get_metadata(context),
        )
    
    @abstractmethod
    def _build_template_context(self, context: CodeGenerationContext) -> Dict[str, Any]:
        """Build the template context from the generation context."""
        pass
    
    def _get_dependencies(self, context: CodeGenerationContext) -> List[str]:
        """Get dependencies for this generator."""
        return []
    
    def _get_declarations(self, context: CodeGenerationContext) -> List[str]:
        """Get declarations for this generator."""
        return []
    
    def _get_metadata(self, context: CodeGenerationContext) -> Dict[str, Any]:
        """Get metadata for this generator."""
        return {
            "generator": self.__class__.__name__,
            "template": self._template_name,
        }


class CompositeGenerator(BaseCodeGenerator):
    """Generator that composes multiple sub-generators."""
    
    def __init__(self, template_renderer: TemplateRenderer, generators: List[CodeGenerator]):
        """Initialize with a list of sub-generators."""
        super().__init__(template_renderer)
        self._generators = generators
    
    def generate(self, context: CodeGenerationContext) -> CodeFragment:
        """Generate code by composing all sub-generators."""
        fragments = []
        all_dependencies = []
        all_declarations = []
        all_metadata = {}
        
        for generator in self._generators:
            fragment = generator.generate(context)
            fragments.append(fragment)
            all_dependencies.extend(fragment.dependencies)
            all_declarations.extend(fragment.declarations)
            all_metadata.update(fragment.metadata)
        
        # Combine all content
        combined_content = "\n\n".join(fragment.content for fragment in fragments)
        
        return self._create_fragment(
            content=combined_content,
            dependencies=list(set(all_dependencies)),  # Remove duplicates
            declarations=list(set(all_declarations)),  # Remove duplicates
            metadata={
                **all_metadata,
                "composite_generator": True,
                "sub_generators": len(self._generators),
            },
        )


def validate_generator_output(fragment: CodeFragment) -> bool:
    """Validate the output of a code generator."""
    if not fragment.content.strip():
        return False
    
    # Check for basic syntax issues
    if fragment.content.count('{') != fragment.content.count('}'):
        return False
    
    if fragment.content.count('(') != fragment.content.count(')'):
        return False
    
    return True
