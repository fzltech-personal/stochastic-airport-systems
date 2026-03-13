"""
Path resolution utilities.

Provides reliable, absolute paths to project resources (configs, data)
regardless of the execution context (IDE, CLI, Docker).
"""
import os
from pathlib import Path
from typing import Optional, List


class ProjectPaths:
    """
    Singleton-like access to project directory structure.
    
    Dynamically resolves the project root by searching for anchor files
    or using an environment variable override.
    """
    
    _root: Optional[Path] = None
    
    # Files that identify the project root
    ANCHOR_FILES: List[str] = [
        "pyproject.toml", 
        "setup.py", 
        ".git", 
        "requirements.txt",
        "README.md"
    ]

    @classmethod
    def get_root(cls) -> Path:
        """
        Get the absolute path to the project root.
        
        Resolution Strategy:
        1. Check PROJECT_ROOT environment variable.
        2. Walk up from the current file (__file__) looking for anchor files.
        3. Fallback to current working directory (cwd).
        
        Returns:
            Path object pointing to the repository root.
            
        Raises:
            RuntimeError: If the project root cannot be determined.
        """
        if cls._root is not None:
            return cls._root

        # 1. Environment Variable Override
        env_root = os.environ.get("PROJECT_ROOT")
        if env_root:
            path = Path(env_root).resolve()
            if path.exists():
                cls._root = path
                return cls._root

        # 2. Walk up from this file
        current_path = Path(__file__).resolve().parent
        for parent in [current_path] + list(current_path.parents):
            for anchor in cls.ANCHOR_FILES:
                if (parent / anchor).exists():
                    cls._root = parent
                    return cls._root

        # 3. Fallback to CWD (useful for interactive shells/notebooks where __file__ might be weird)
        # Note: This is less safe but acceptable as a last resort
        cwd = Path.cwd().resolve()
        for anchor in cls.ANCHOR_FILES:
            if (cwd / anchor).exists():
                cls._root = cwd
                return cls._root
        
        # If we reached here, we are lost.
        # Defaulting to CWD might be dangerous if we are in a subdirectory, 
        # but it's better than crashing immediately in some scripts.
        # Ideally, we should raise an error in a strict environment.
        raise RuntimeError(
            f"Could not find project root. Searched up from {current_path} "
            f"for anchors: {cls.ANCHOR_FILES}"
        )

    @classmethod
    def get_configs_dir(cls) -> Path:
        """Get absolute path to configs directory."""
        return cls.get_root() / "configs"

    @classmethod
    def get_data_dir(cls) -> Path:
        """Get absolute path to data directory."""
        return cls.get_root() / "data"

    @classmethod
    def get_src_dir(cls) -> Path:
        """Get absolute path to source code directory."""
        return cls.get_root() / "src"
    
    @classmethod
    def resolve_path(cls, path_str: str) -> Path:
        """
        Resolve a potentially relative path against the project root.
        
        Args:
            path_str: String representation of a path.
            
        Returns:
            Absolute Path object.
        """
        path = Path(path_str)
        if path.is_absolute():
            return path
        return cls.get_root() / path
