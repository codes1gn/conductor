"""
Header Generator for DSL Files.

This module provides generators for creating DSL file headers,
including includes, function signatures, and initial declarations.
"""

from __future__ import annotations

from typing import Dict, Any, List
from dataclasses import dataclass

from ..types import (
    CodeGenerationContext,
    CodeFragment,
    TemplateRenderer,
    BufferInfo,
)
from .base import TemplateBasedGenerator


@dataclass(frozen=True)
class HeaderConfig:
    """Configuration for header generation."""
    include_choreo: bool = True
    include_custom: List[str] = None
    add_comments: bool = True
    function_prefix: str = "__co__ auto"
    
    def __post_init__(self):
        if self.include_custom is None:
            object.__setattr__(self, 'include_custom', [])


class HeaderGenerator(TemplateBasedGenerator):
    """Generator for DSL file headers."""
    
    def __init__(self, template_renderer: TemplateRenderer, config: HeaderConfig = None):
        """Initialize the header generator."""
        super().__init__(template_renderer, "header.j2")
        self._config = config or HeaderConfig()
    
    def _build_template_context(self, context: CodeGenerationContext) -> Dict[str, Any]:
        """Build template context for header generation."""
        return {
            "include_choreo": self._config.include_choreo,
            "custom_includes": self._config.include_custom,
            "add_comments": self._config.add_comments,
            "function_name": context.function_name,
            "generation_config": context.config,
        }
    
    def _get_dependencies(self, context: CodeGenerationContext) -> List[str]:
        """Get header dependencies."""
        deps = []
        if self._config.include_choreo:
            deps.append("choreo.h")
        deps.extend(self._config.include_custom)
        return deps


class FunctionSignatureGenerator(TemplateBasedGenerator):
    """Generator for function signatures."""
    
    def __init__(self, template_renderer: TemplateRenderer, config: HeaderConfig = None):
        """Initialize the function signature generator."""
        super().__init__(template_renderer, "function_signature.j2")
        self._config = config or HeaderConfig()
    
    def _build_template_context(self, context: CodeGenerationContext) -> Dict[str, Any]:
        """Build template context for function signature."""
        # Convert BufferInfo to parameter format
        input_params = []
        for buffer in context.input_buffers:
            param = {
                "name": buffer.name,
                "dtype": buffer.dtype,
                "shape": buffer.shape,
                "is_input": True,
            }
            input_params.append(param)
        
        output_params = []
        for buffer in context.output_buffers:
            param = {
                "name": buffer.name,
                "dtype": buffer.dtype,
                "shape": buffer.shape,
                "is_output": True,
            }
            output_params.append(param)
        
        return {
            "function_name": context.function_name,
            "function_prefix": self._config.function_prefix,
            "input_params": input_params,
            "output_params": output_params,
            "all_params": input_params + output_params,
        }


class DeclarationGenerator(TemplateBasedGenerator):
    """Generator for variable declarations."""
    
    def __init__(self, template_renderer: TemplateRenderer):
        """Initialize the declaration generator."""
        super().__init__(template_renderer, "declarations.j2")
    
    def _build_template_context(self, context: CodeGenerationContext) -> Dict[str, Any]:
        """Build template context for declarations."""
        # Generate local buffer declarations
        local_buffers = []
        
        # Add intermediate buffers based on the computation
        # This would be populated by analyzing the DAG
        for i, output_buffer in enumerate(context.output_buffers):
            local_buffer = {
                "name": f"local_{output_buffer.name}",
                "dtype": output_buffer.dtype,
                "shape": output_buffer.shape,
                "memory_level": context.config.memory_level.value,
            }
            local_buffers.append(local_buffer)
        
        return {
            "local_buffers": local_buffers,
            "memory_level": context.config.memory_level.value,
            "parallel_config": context.parallel_config,
        }


class CompleteHeaderGenerator:
    """Generator that combines header, signature, and declarations."""
    
    def __init__(self, template_renderer: TemplateRenderer, config: HeaderConfig = None):
        """Initialize the complete header generator."""
        self._template_renderer = template_renderer
        self._config = config or HeaderConfig()
        
        # Create sub-generators
        self._header_gen = HeaderGenerator(template_renderer, config)
        self._signature_gen = FunctionSignatureGenerator(template_renderer, config)
        self._declaration_gen = DeclarationGenerator(template_renderer)
    
    def generate(self, context: CodeGenerationContext) -> CodeFragment:
        """Generate complete header including includes, signature, and declarations."""
        # Generate each component
        header_fragment = self._header_gen.generate(context)
        signature_fragment = self._signature_gen.generate(context)
        declaration_fragment = self._declaration_gen.generate(context)
        
        # Combine into complete header
        complete_content = "\n".join([
            header_fragment.content,
            "",  # Empty line
            signature_fragment.content,
            declaration_fragment.content,
        ])
        
        # Combine metadata
        combined_metadata = {
            **header_fragment.metadata,
            **signature_fragment.metadata,
            **declaration_fragment.metadata,
            "complete_header": True,
            "components": ["header", "signature", "declarations"],
        }
        
        # Combine dependencies and declarations
        all_dependencies = list(set(
            list(header_fragment.dependencies) +
            list(signature_fragment.dependencies) +
            list(declaration_fragment.dependencies)
        ))
        
        all_declarations = list(set(
            list(header_fragment.declarations) +
            list(signature_fragment.declarations) +
            list(declaration_fragment.declarations)
        ))
        
        return CodeFragment(
            content=complete_content,
            dependencies=tuple(all_dependencies),
            declarations=tuple(all_declarations),
            metadata=combined_metadata,
        )
