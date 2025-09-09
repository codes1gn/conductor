"""
Type Mapping Utilities for Conductor Framework.

This module provides comprehensive type mapping between PyTorch, DAG, and Choreo
type systems, centralizing all type conversion logic in one place.
"""

from __future__ import annotations

import torch
from typing import Dict, Optional, Union, Any, Type
from enum import Enum
from dataclasses import dataclass

from .logging import get_logger

logger = get_logger(__name__)


class ChoreoDataType(Enum):
    """Choreo DSL data types."""
    
    # Floating point types
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    F64 = "f64"
    
    # Signed integer types
    S8 = "s8"
    S16 = "s16"
    S32 = "s32"
    S64 = "s64"
    
    # Unsigned integer types
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    
    # Boolean type
    BOOL = "bool"
    
    # Complex types
    C64 = "c64"  # Complex float32
    C128 = "c128"  # Complex float64


@dataclass
class TypeInfo:
    """Information about a data type."""
    
    torch_dtype: torch.dtype
    choreo_type: ChoreoDataType
    size_bytes: int
    is_floating: bool
    is_signed: bool
    description: str


class TypeMapper:
    """
    Comprehensive type mapper between PyTorch, DAG, and Choreo type systems.
    
    This class provides bidirectional mapping and validation for all supported
    data types across the different representations used in the framework.
    """
    
    def __init__(self):
        """Initialize the type mapper with comprehensive mappings."""
        self._torch_to_choreo = self._build_torch_to_choreo_map()
        self._choreo_to_torch = self._build_choreo_to_torch_map()
        self._string_to_choreo = self._build_string_to_choreo_map()
        self._type_info = self._build_type_info_map()
    
    def _build_torch_to_choreo_map(self) -> Dict[torch.dtype, ChoreoDataType]:
        """Build mapping from PyTorch dtypes to Choreo types."""
        return {
            torch.float32: ChoreoDataType.F32,
            torch.float16: ChoreoDataType.F16,
            torch.bfloat16: ChoreoDataType.BF16,
            torch.float64: ChoreoDataType.F64,
            torch.int8: ChoreoDataType.S8,
            torch.int16: ChoreoDataType.S16,
            torch.int32: ChoreoDataType.S32,
            torch.int64: ChoreoDataType.S64,
            torch.uint8: ChoreoDataType.U8,
            torch.bool: ChoreoDataType.BOOL,
            torch.complex64: ChoreoDataType.C64,
            torch.complex128: ChoreoDataType.C128,
        }
    
    def _build_choreo_to_torch_map(self) -> Dict[ChoreoDataType, torch.dtype]:
        """Build mapping from Choreo types to PyTorch dtypes."""
        return {v: k for k, v in self._torch_to_choreo.items()}
    
    def _build_string_to_choreo_map(self) -> Dict[str, ChoreoDataType]:
        """Build mapping from string representations to Choreo types."""
        return {
            # PyTorch string representations
            "torch.float32": ChoreoDataType.F32,
            "torch.float16": ChoreoDataType.F16,
            "torch.bfloat16": ChoreoDataType.BF16,
            "torch.float64": ChoreoDataType.F64,
            "torch.int8": ChoreoDataType.S8,
            "torch.int16": ChoreoDataType.S16,
            "torch.int32": ChoreoDataType.S32,
            "torch.int64": ChoreoDataType.S64,
            "torch.uint8": ChoreoDataType.U8,
            "torch.bool": ChoreoDataType.BOOL,
            "torch.complex64": ChoreoDataType.C64,
            "torch.complex128": ChoreoDataType.C128,
            
            # Direct Choreo string representations
            "f32": ChoreoDataType.F32,
            "f16": ChoreoDataType.F16,
            "bf16": ChoreoDataType.BF16,
            "f64": ChoreoDataType.F64,
            "s8": ChoreoDataType.S8,
            "s16": ChoreoDataType.S16,
            "s32": ChoreoDataType.S32,
            "s64": ChoreoDataType.S64,
            "u8": ChoreoDataType.U8,
            "u16": ChoreoDataType.U16,
            "u32": ChoreoDataType.U32,
            "u64": ChoreoDataType.U64,
            "bool": ChoreoDataType.BOOL,
            "c64": ChoreoDataType.C64,
            "c128": ChoreoDataType.C128,
        }
    
    def _build_type_info_map(self) -> Dict[ChoreoDataType, TypeInfo]:
        """Build comprehensive type information map."""
        return {
            ChoreoDataType.F32: TypeInfo(
                torch.float32, ChoreoDataType.F32, 4, True, True, "32-bit floating point"
            ),
            ChoreoDataType.F16: TypeInfo(
                torch.float16, ChoreoDataType.F16, 2, True, True, "16-bit floating point"
            ),
            ChoreoDataType.BF16: TypeInfo(
                torch.bfloat16, ChoreoDataType.BF16, 2, True, True, "16-bit brain floating point"
            ),
            ChoreoDataType.F64: TypeInfo(
                torch.float64, ChoreoDataType.F64, 8, True, True, "64-bit floating point"
            ),
            ChoreoDataType.S8: TypeInfo(
                torch.int8, ChoreoDataType.S8, 1, False, True, "8-bit signed integer"
            ),
            ChoreoDataType.S16: TypeInfo(
                torch.int16, ChoreoDataType.S16, 2, False, True, "16-bit signed integer"
            ),
            ChoreoDataType.S32: TypeInfo(
                torch.int32, ChoreoDataType.S32, 4, False, True, "32-bit signed integer"
            ),
            ChoreoDataType.S64: TypeInfo(
                torch.int64, ChoreoDataType.S64, 8, False, True, "64-bit signed integer"
            ),
            ChoreoDataType.U8: TypeInfo(
                torch.uint8, ChoreoDataType.U8, 1, False, False, "8-bit unsigned integer"
            ),
            ChoreoDataType.BOOL: TypeInfo(
                torch.bool, ChoreoDataType.BOOL, 1, False, False, "Boolean type"
            ),
            ChoreoDataType.C64: TypeInfo(
                torch.complex64, ChoreoDataType.C64, 8, True, True, "64-bit complex (2x32-bit float)"
            ),
            ChoreoDataType.C128: TypeInfo(
                torch.complex128, ChoreoDataType.C128, 16, True, True, "128-bit complex (2x64-bit float)"
            ),
        }
    
    def torch_to_choreo(self, torch_dtype: torch.dtype) -> ChoreoDataType:
        """
        Convert PyTorch dtype to Choreo type.
        
        Args:
            torch_dtype: PyTorch data type
            
        Returns:
            Corresponding Choreo data type
            
        Raises:
            ValueError: If dtype is not supported
        """
        if torch_dtype not in self._torch_to_choreo:
            raise ValueError(f"Unsupported PyTorch dtype: {torch_dtype}")
        return self._torch_to_choreo[torch_dtype]
    
    def choreo_to_torch(self, choreo_type: Union[ChoreoDataType, str]) -> torch.dtype:
        """
        Convert Choreo type to PyTorch dtype.
        
        Args:
            choreo_type: Choreo data type (enum or string)
            
        Returns:
            Corresponding PyTorch data type
            
        Raises:
            ValueError: If type is not supported
        """
        if isinstance(choreo_type, str):
            choreo_type = self.string_to_choreo(choreo_type)
        
        if choreo_type not in self._choreo_to_torch:
            raise ValueError(f"Unsupported Choreo type: {choreo_type}")
        return self._choreo_to_torch[choreo_type]
    
    def string_to_choreo(self, type_str: str) -> ChoreoDataType:
        """
        Convert string representation to Choreo type.
        
        Args:
            type_str: String representation of type
            
        Returns:
            Corresponding Choreo data type
            
        Raises:
            ValueError: If string is not recognized
        """
        if type_str not in self._string_to_choreo:
            raise ValueError(f"Unrecognized type string: {type_str}")
        return self._string_to_choreo[type_str]
    
    def torch_to_choreo_string(self, torch_dtype: torch.dtype) -> str:
        """
        Convert PyTorch dtype directly to Choreo string.
        
        Args:
            torch_dtype: PyTorch data type
            
        Returns:
            Choreo type string
        """
        choreo_type = self.torch_to_choreo(torch_dtype)
        return choreo_type.value
    
    def get_type_info(self, choreo_type: Union[ChoreoDataType, str]) -> TypeInfo:
        """
        Get comprehensive information about a type.
        
        Args:
            choreo_type: Choreo data type (enum or string)
            
        Returns:
            Type information
        """
        if isinstance(choreo_type, str):
            choreo_type = self.string_to_choreo(choreo_type)
        
        if choreo_type not in self._type_info:
            raise ValueError(f"No type info available for: {choreo_type}")
        return self._type_info[choreo_type]
    
    def is_supported_torch_dtype(self, torch_dtype: torch.dtype) -> bool:
        """Check if PyTorch dtype is supported."""
        return torch_dtype in self._torch_to_choreo
    
    def is_supported_choreo_type(self, choreo_type: Union[ChoreoDataType, str]) -> bool:
        """Check if Choreo type is supported."""
        if isinstance(choreo_type, str):
            return choreo_type in self._string_to_choreo
        return choreo_type in self._choreo_to_torch
    
    def get_supported_torch_dtypes(self) -> list[torch.dtype]:
        """Get list of all supported PyTorch dtypes."""
        return list(self._torch_to_choreo.keys())
    
    def get_supported_choreo_types(self) -> list[ChoreoDataType]:
        """Get list of all supported Choreo types."""
        return list(self._choreo_to_torch.keys())


# Global type mapper instance
_global_type_mapper: Optional[TypeMapper] = None


def get_type_mapper() -> TypeMapper:
    """Get the global type mapper instance."""
    global _global_type_mapper
    if _global_type_mapper is None:
        _global_type_mapper = TypeMapper()
    return _global_type_mapper


# Convenience functions for common operations
def torch_to_choreo_string(torch_dtype: torch.dtype) -> str:
    """Convert PyTorch dtype to Choreo string."""
    return get_type_mapper().torch_to_choreo_string(torch_dtype)


def choreo_to_torch_dtype(choreo_type: Union[ChoreoDataType, str]) -> torch.dtype:
    """Convert Choreo type to PyTorch dtype."""
    return get_type_mapper().choreo_to_torch(choreo_type)


def is_supported_dtype(dtype: Union[torch.dtype, str]) -> bool:
    """Check if a dtype is supported."""
    mapper = get_type_mapper()
    if isinstance(dtype, torch.dtype):
        return mapper.is_supported_torch_dtype(dtype)
    else:
        return mapper.is_supported_choreo_type(dtype)


def get_dtype_size_bytes(dtype: Union[torch.dtype, ChoreoDataType, str]) -> int:
    """Get size in bytes for a data type."""
    mapper = get_type_mapper()
    if isinstance(dtype, torch.dtype):
        choreo_type = mapper.torch_to_choreo(dtype)
    elif isinstance(dtype, str):
        choreo_type = mapper.string_to_choreo(dtype)
    else:
        choreo_type = dtype
    
    type_info = mapper.get_type_info(choreo_type)
    return type_info.size_bytes
