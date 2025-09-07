"""
Custom exception definitions.

This module defines the exception hierarchy for Conductor-specific
errors and provides fallback handling mechanisms.
"""

import torch
from typing import Optional, Callable
from ..config.logging import get_logger

logger = get_logger(__name__)


class ConductorError(Exception):
    """
    Base exception for all Conductor-related errors.
    
    This is the root exception class for all Conductor-specific
    errors, providing common functionality and error handling.
    """
    
    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Initialize Conductor error.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class CompilationError(ConductorError):
    """
    Raised when DSL compilation fails.
    
    This exception is raised when the Conductor compiler fails to
    compile generated DSL code into executable artifacts.
    """
    
    def __init__(self, message: str, dsl_code: str = "", compiler_output: str = ""):
        """
        Initialize compilation error.
        
        Args:
            message: Error description
            dsl_code: DSL code that failed to compile
            compiler_output: Output from the compiler
        """
        details = {}
        if dsl_code:
            details['dsl_length'] = len(dsl_code)
        if compiler_output:
            details['compiler_output_length'] = len(compiler_output)
            
        super().__init__(message, details)
        self.dsl_code = dsl_code
        self.compiler_output = compiler_output
        
    def get_compiler_errors(self) -> list:
        """
        Extract error messages from compiler output.
        
        Returns:
            List of error message strings
        """
        if not self.compiler_output:
            return []
            
        # Simple error extraction (can be enhanced)
        lines = self.compiler_output.split('\n')
        errors = []
        for line in lines:
            if 'error:' in line.lower() or 'failed:' in line.lower():
                errors.append(line.strip())
        return errors


class UnsupportedOperationError(ConductorError):
    """
    Raised when an operation cannot be converted to Conductor DSL.
    
    This exception indicates that a PyTorch operation is not supported
    by the Conductor backend and requires fallback handling.
    """
    
    def __init__(self, operation: str, reason: str = ""):
        """
        Initialize unsupported operation error.
        
        Args:
            operation: Name of the unsupported operation
            reason: Optional explanation for why it's unsupported
        """
        message = f"Unsupported operation '{operation}'"
        if reason:
            message += f": {reason}"
            
        super().__init__(message, {'operation': operation, 'reason': reason})
        self.operation = operation
        self.reason = reason


class DeviceError(ConductorError):
    """
    Raised when GCU device operations fail.
    
    This exception covers device initialization, memory allocation,
    and kernel execution failures.
    """
    
    def __init__(self, message: str, device_id: Optional[int] = None):
        """
        Initialize device error.
        
        Args:
            message: Error description
            device_id: Optional device identifier
        """
        details = {}
        if device_id is not None:
            details['device_id'] = device_id
            
        super().__init__(message, details)
        self.device_id = device_id


class ExecutionError(ConductorError):
    """
    Raised when kernel execution fails on GCU hardware.

    This exception is raised when a compiled kernel fails to execute
    properly on the GCU device, including runtime errors and device failures.
    """

    def __init__(self, message: str, kernel_name: Optional[str] = None, exit_code: Optional[int] = None):
        """
        Initialize execution error.

        Args:
            message: Error description
            kernel_name: Optional name of the failing kernel
            exit_code: Optional kernel exit code
        """
        details = {}
        if kernel_name is not None:
            details['kernel_name'] = kernel_name
        if exit_code is not None:
            details['exit_code'] = exit_code

        super().__init__(message, details)
        self.kernel_name = kernel_name
        self.exit_code = exit_code


class FallbackHandler:
    """
    Manages graceful fallback to alternative backends.
    
    This class implements the logic for determining when to fallback
    and executing fallback compilation using the Inductor backend.
    """
    
    def __init__(self):
        """Initialize fallback handler."""
        self._fallback_count = 0
        self._fallback_reasons = {}
        
    def should_fallback(self, error: Exception) -> bool:
        """
        Determine if error should trigger fallback mechanism.
        
        Args:
            error: Exception that occurred during compilation
            
        Returns:
            True if fallback should be attempted, False otherwise
        """
        # Always fallback for unsupported operations
        if isinstance(error, UnsupportedOperationError):
            return True
            
        # Fallback for compilation errors (may be temporary)
        if isinstance(error, CompilationError):
            return True
            
        # Fallback for device errors
        if isinstance(error, DeviceError):
            return True
            
        # Don't fallback for other types of errors
        return False
        
    def execute_fallback(self, graph_module: torch.fx.GraphModule, reason: str = "") -> Callable:
        """
        Execute fallback compilation using Inductor backend.
        
        Args:
            graph_module: FX Graph to compile with fallback backend
            reason: Reason for fallback (for logging)
            
        Returns:
            Compiled function using fallback backend
            
        Raises:
            RuntimeError: If fallback compilation also fails
        """
        self._fallback_count += 1
        
        # Track fallback reasons for debugging
        if reason:
            self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1
            
        logger.warning(f"Falling back to Inductor backend (reason: {reason})")
        
        try:
            # Use PyTorch's default Inductor backend
            if hasattr(torch, 'compile'):
                return torch.compile(graph_module, backend='inductor')
            else:
                # Fallback for older PyTorch versions
                return graph_module.forward
                
        except AttributeError:
            # torch.compile not available, use eager execution
            logger.warning("torch.compile not available, using eager execution")
            return graph_module.forward
            
        except Exception as e:
            raise RuntimeError(f"Fallback compilation failed: {e}")
            
    def get_fallback_stats(self) -> dict:
        """
        Get statistics about fallback usage.
        
        Returns:
            Dictionary containing fallback statistics
        """
        return {
            'total_fallbacks': self._fallback_count,
            'fallback_reasons': dict(self._fallback_reasons),
            'most_common_reason': max(self._fallback_reasons.items(), 
                                    key=lambda x: x[1])[0] if self._fallback_reasons else None
        }
        
    def reset_stats(self) -> None:
        """Reset fallback statistics."""
        self._fallback_count = 0
        self._fallback_reasons.clear()


# Global fallback handler instance
_fallback_handler = FallbackHandler()

def get_fallback_handler() -> FallbackHandler:
    """Get the global fallback handler instance."""
    return _fallback_handler