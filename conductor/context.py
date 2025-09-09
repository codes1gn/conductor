"""
Centralized Context Management for Conductor.

This module provides a context manager that handles all global registries,
generators, and singleton objects with proper lifecycle management.
"""

from __future__ import annotations

import threading
from typing import Optional, Any, Dict
from contextlib import contextmanager
from dataclasses import dataclass, field

from .utils.logging import get_logger
from .utils.config import get_config

logger = get_logger(__name__)


@dataclass
class ConductorContext:
    """
    Centralized context manager for all Conductor global state.
    
    Manages lifecycle of registries, generators, and singleton objects
    with proper initialization and cleanup.
    """
    
    # Registry instances
    operator_registry: Optional[Any] = None
    device_kernel_registry: Optional[Any] = None
    operation_handler_registry: Optional[Any] = None
    registry_template_engine: Optional[Any] = None
    signature_registry: Optional[Any] = None
    
    # Generator instances
    dma_generator: Optional[Any] = None
    modern_dsl_generator: Optional[Any] = None
    default_identifier_generator: Optional[Any] = None
    
    # Manager instances
    default_buffer_naming_manager: Optional[Any] = None
    debug_artifact_manager: Optional[Any] = None
    
    # Operation factory
    operation_factory: Optional[Any] = None
    
    # Context state
    _initialized: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    def __enter__(self) -> 'ConductorContext':
        """Enter context and initialize all global state."""
        with self._lock:
            if not self._initialized:
                self._initialize_all_components()
                self._initialized = True
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and cleanup resources."""
        with self._lock:
            if self._initialized:
                self._cleanup_all_components()
                self._initialized = False
    
    def _initialize_all_components(self) -> None:
        """Initialize all global components in proper order."""
        logger.info("Initializing Conductor context...")
        
        # Initialize configuration first
        config = get_config()
        
        # Initialize core registries
        self._initialize_operator_registry()
        self._initialize_device_kernel_registry()
        self._initialize_operation_handler_registry()
        self._initialize_signature_registry()
        
        # Initialize generators
        self._initialize_generators()
        
        # Initialize managers
        self._initialize_managers()
        
        # Initialize template engine (depends on registries)
        self._initialize_template_engine()
        
        # Initialize operation factory (depends on registries)
        self._initialize_operation_factory()
        
        logger.info("Conductor context initialization complete")
    
    def _initialize_operator_registry(self) -> None:
        """Initialize the operator registry."""
        from .codegen.operator_registry import OperatorRegistry
        self.operator_registry = OperatorRegistry()
        logger.debug("Initialized operator registry")
    
    def _initialize_device_kernel_registry(self) -> None:
        """Initialize the device kernel registry."""
        from .codegen.device_kernels import DeviceKernelRegistry
        self.device_kernel_registry = DeviceKernelRegistry()
        logger.debug("Initialized device kernel registry")
    
    def _initialize_operation_handler_registry(self) -> None:
        """Initialize the operation handler registry."""
        # Operation handler registry has been removed in favor of unified operation factory
        self.operation_handler_registry = None
        logger.debug("Operation handler registry deprecated - using operation factory")
    
    def _initialize_signature_registry(self) -> None:
        """Initialize the signature registry."""
        from .codegen.function_signature import SignatureRegistry
        self.signature_registry = SignatureRegistry()
        logger.debug("Initialized signature registry")
    
    def _initialize_generators(self) -> None:
        """Initialize all generator instances."""
        from .codegen.dma_generator import DMAGenerator
        from .codegen.dsl_constants import DSLIdentifierGenerator

        self.dma_generator = DMAGenerator()
        self.modern_dsl_generator = None  # Removed - using unified ChoreoDslGen
        self.default_identifier_generator = DSLIdentifierGenerator()
        logger.debug("Initialized generators")
    
    def _initialize_managers(self) -> None:
        """Initialize all manager instances."""
        from .utils.naming import BufferNamingManager
        from .utils.artifacts import DebugArtifactManager
        
        self.default_buffer_naming_manager = BufferNamingManager()
        self.debug_artifact_manager = DebugArtifactManager()
        logger.debug("Initialized managers")
    
    def _initialize_template_engine(self) -> None:
        """Initialize the template engine."""
        from .codegen.registry_based_templates import RegistryBasedTemplateEngine
        self.registry_template_engine = RegistryBasedTemplateEngine()
        logger.debug("Initialized template engine")
    
    def _initialize_operation_factory(self) -> None:
        """Initialize the operation factory."""
        # Skip operation factory initialization to avoid circular dependencies
        # It will be initialized lazily when needed
        self.operation_factory = None
        logger.debug("Operation factory will be initialized lazily")
    
    def _cleanup_all_components(self) -> None:
        """Cleanup all components."""
        logger.info("Cleaning up Conductor context...")
        
        # Reset all instances to None
        self.operator_registry = None
        self.device_kernel_registry = None
        self.operation_handler_registry = None
        self.registry_template_engine = None
        self.signature_registry = None
        self.dma_generator = None
        self.modern_dsl_generator = None
        self.default_identifier_generator = None
        self.default_buffer_naming_manager = None
        self.debug_artifact_manager = None
        self.operation_factory = None
        
        logger.info("Conductor context cleanup complete")
    
    def get_operator_registry(self):
        """Get the operator registry instance."""
        if not self._initialized:
            raise RuntimeError("Context not initialized. Use 'with ConductorContext()' or call __enter__().")
        return self.operator_registry
    
    def get_device_kernel_registry(self):
        """Get the device kernel registry instance."""
        if not self._initialized:
            raise RuntimeError("Context not initialized. Use 'with ConductorContext()' or call __enter__().")
        return self.device_kernel_registry
    
    def get_dma_generator(self):
        """Get the DMA generator instance."""
        if not self._initialized:
            raise RuntimeError("Context not initialized. Use 'with ConductorContext()' or call __enter__().")
        return self.dma_generator
    
    def get_operation_factory(self):
        """Get the operation factory instance."""
        if not self._initialized:
            raise RuntimeError("Context not initialized. Use 'with ConductorContext()' or call __enter__().")

        # Lazy initialization to avoid circular dependencies
        if self.operation_factory is None:
            try:
                from .codegen.operation_factory import OperationFactory
                self.operation_factory = OperationFactory()
                logger.debug("Lazily initialized operation factory")
            except Exception as e:
                logger.error(f"Failed to initialize operation factory: {e}")
                raise

        return self.operation_factory


# Global context instance
_global_context: Optional[ConductorContext] = None
_context_lock = threading.Lock()


def get_global_context() -> ConductorContext:
    """Get or create the global context instance."""
    global _global_context
    with _context_lock:
        if _global_context is None:
            _global_context = ConductorContext()
        return _global_context


@contextmanager
def conductor_context():
    """Context manager for Conductor operations."""
    context = get_global_context()
    with context:
        yield context


def ensure_context_initialized() -> ConductorContext:
    """Ensure the global context is initialized and return it."""
    context = get_global_context()
    if not context._initialized:
        context.__enter__()
    return context
