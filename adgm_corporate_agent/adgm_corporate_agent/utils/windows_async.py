"""
Windows-compatible async file operations wrapper.
Replaces aiofiles functionality for Windows systems.
"""

import asyncio
import io
from pathlib import Path
from typing import Union, Optional, Any

class AsyncFileWrapper:
    """Windows-compatible async file wrapper."""
    
    def __init__(self, file_path: Union[str, Path], mode: str = 'r', encoding: Optional[str] = None):
        self.file_path = Path(file_path)
        self.mode = mode
        self.encoding = encoding
    
    async def read(self) -> Union[str, bytes]:
        """Read file content asynchronously."""
        loop = asyncio.get_event_loop()
        
        def _read():
            if 'b' in self.mode:
                with open(self.file_path, self.mode) as f:
                    return f.read()
            else:
                with open(self.file_path, self.mode, encoding=self.encoding or 'utf-8') as f:
                    return f.read()
        
        return await loop.run_in_executor(None, _read)
    
    async def write(self, content: Union[str, bytes]) -> None:
        """Write content to file asynchronously."""
        loop = asyncio.get_event_loop()
        
        def _write():
            if 'b' in self.mode:
                with open(self.file_path, self.mode) as f:
                    f.write(content)
            else:
                with open(self.file_path, self.mode, encoding=self.encoding or 'utf-8') as f:
                    f.write(content)
        
        await loop.run_in_executor(None, _write)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

def open_async(file_path: Union[str, Path], mode: str = 'r', encoding: Optional[str] = None):
    """Async file opener compatible with Windows."""
    return AsyncFileWrapper(file_path, mode, encoding)

# Compatibility functions
async def read_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """Read file asynchronously."""
    wrapper = AsyncFileWrapper(file_path, 'r', encoding)
    return await wrapper.read()

async def write_file(file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """Write file asynchronously."""
    wrapper = AsyncFileWrapper(file_path, 'w', encoding)
    await wrapper.write(content)

async def read_bytes(file_path: Union[str, Path]) -> bytes:
    """Read file as bytes asynchronously."""
    wrapper = AsyncFileWrapper(file_path, 'rb')
    return await wrapper.read()

async def write_bytes(file_path: Union[str, Path], content: bytes) -> None:
    """Write bytes to file asynchronously."""
    wrapper = AsyncFileWrapper(file_path, 'wb')
    await wrapper.write(content)
