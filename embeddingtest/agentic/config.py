"""
Agentic Configuration System

Easy configuration for enterprise deployments. Configure via:
1. Environment variables (AGENTIC_*)
2. Config file (.agentic/config.json or agentic.config.json)
3. Direct code configuration

Priority: Direct code > Environment variables > Config file > Defaults

Example config file (agentic.config.json):
{
    "disable_semantic_search": true,
    "llm_provider": "custom"
}

Example environment variables:
    AGENTIC_DISABLE_SEMANTIC_SEARCH=true
    AGENTIC_LLM_PROVIDER=custom
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgenticConfig:
    """
    Configuration for Agentic system.
    
    Attributes:
        disable_semantic_search: If True, disables the semantic_search tool.
            Set this to True if you can't run embedding models.
            Default: False (semantic search enabled if ChromaDB available)
        
        llm_provider: The LLM provider to use. Options:
            - "openrouter" (default): Use OpenRouter API
            - "custom": Use a custom LLM client provided via code
        
        openrouter_api_key: API key for OpenRouter (if using openrouter provider)
        
        openrouter_model: Model to use with OpenRouter
        
        max_tool_rounds: Maximum tool call rounds before stopping
        
        context_limit: Token context limit for the model
    """
    # Core feature toggles
    disable_semantic_search: bool = False
    
    # LLM configuration
    llm_provider: str = "openrouter"  # "openrouter" or "custom"
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-sonnet-4.5"
    
    # Execution settings
    max_tool_rounds: int = 20
    context_limit: int = 200000
    max_execution_attempts: int = 3
    
    # Debug settings
    debug_logging: bool = True
    
    @classmethod
    def from_env(cls) -> 'AgenticConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Feature toggles
        if os.getenv("AGENTIC_DISABLE_SEMANTIC_SEARCH"):
            config.disable_semantic_search = os.getenv("AGENTIC_DISABLE_SEMANTIC_SEARCH", "").lower() in ("true", "1", "yes")
        
        # LLM settings
        if os.getenv("AGENTIC_LLM_PROVIDER"):
            config.llm_provider = os.getenv("AGENTIC_LLM_PROVIDER")
        
        if os.getenv("OPENROUTER_API_KEY"):
            config.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if os.getenv("AGENTIC_MODEL"):
            config.openrouter_model = os.getenv("AGENTIC_MODEL")
        
        # Execution settings
        if os.getenv("AGENTIC_MAX_TOOL_ROUNDS"):
            try:
                config.max_tool_rounds = int(os.getenv("AGENTIC_MAX_TOOL_ROUNDS"))
            except ValueError:
                pass
        
        if os.getenv("AGENTIC_CONTEXT_LIMIT"):
            try:
                config.context_limit = int(os.getenv("AGENTIC_CONTEXT_LIMIT"))
            except ValueError:
                pass
        
        return config
    
    @classmethod
    def from_file(cls, path: Path) -> 'AgenticConfig':
        """Load configuration from a JSON file."""
        config = cls()
        
        if not path.exists():
            return config
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Feature toggles
            if "disable_semantic_search" in data:
                config.disable_semantic_search = bool(data["disable_semantic_search"])
            
            # LLM settings
            if "llm_provider" in data:
                config.llm_provider = str(data["llm_provider"])
            
            if "openrouter_api_key" in data:
                config.openrouter_api_key = str(data["openrouter_api_key"])
            
            if "openrouter_model" in data:
                config.openrouter_model = str(data["openrouter_model"])
            
            if "model" in data:  # alias
                config.openrouter_model = str(data["model"])
            
            # Execution settings
            if "max_tool_rounds" in data:
                config.max_tool_rounds = int(data["max_tool_rounds"])
            
            if "context_limit" in data:
                config.context_limit = int(data["context_limit"])
            
            if "max_execution_attempts" in data:
                config.max_execution_attempts = int(data["max_execution_attempts"])
            
            if "debug_logging" in data:
                config.debug_logging = bool(data["debug_logging"])
                
        except (json.JSONDecodeError, ValueError, TypeError):
            # Invalid config file, use defaults
            pass
        
        return config
    
    @classmethod
    def load(cls, repo_path: Optional[Path] = None) -> 'AgenticConfig':
        """
        Load configuration from all sources (file, env, defaults).
        
        Priority: Environment variables > Config file > Defaults
        
        Args:
            repo_path: Path to repository (to look for config files)
        
        Returns:
            Merged configuration
        """
        # Start with defaults
        config = cls()
        
        # Try to load from config file
        if repo_path:
            config_paths = [
                repo_path / "agentic.config.json",
                repo_path / ".agentic" / "config.json",
            ]
            for config_path in config_paths:
                if config_path.exists():
                    config = cls.from_file(config_path)
                    break
        
        # Override with environment variables
        env_config = cls.from_env()
        
        # Merge (env takes priority)
        if os.getenv("AGENTIC_DISABLE_SEMANTIC_SEARCH"):
            config.disable_semantic_search = env_config.disable_semantic_search
        
        if os.getenv("AGENTIC_LLM_PROVIDER"):
            config.llm_provider = env_config.llm_provider
        
        if os.getenv("OPENROUTER_API_KEY"):
            config.openrouter_api_key = env_config.openrouter_api_key
        
        if os.getenv("AGENTIC_MODEL"):
            config.openrouter_model = env_config.openrouter_model
        
        if os.getenv("AGENTIC_MAX_TOOL_ROUNDS"):
            config.max_tool_rounds = env_config.max_tool_rounds
        
        if os.getenv("AGENTIC_CONTEXT_LIMIT"):
            config.context_limit = env_config.context_limit
        
        return config
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "disable_semantic_search": self.disable_semantic_search,
            "llm_provider": self.llm_provider,
            "openrouter_model": self.openrouter_model,
            "max_tool_rounds": self.max_tool_rounds,
            "context_limit": self.context_limit,
            "max_execution_attempts": self.max_execution_attempts,
            "debug_logging": self.debug_logging,
        }
    
    def save(self, path: Path):
        """Save configuration to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global config instance (can be overridden)
_global_config: Optional[AgenticConfig] = None


def get_config(repo_path: Optional[Path] = None) -> AgenticConfig:
    """Get the current configuration."""
    global _global_config
    if _global_config is None:
        _global_config = AgenticConfig.load(repo_path)
    return _global_config


def set_config(config: AgenticConfig):
    """Set the global configuration."""
    global _global_config
    _global_config = config


def reset_config():
    """Reset configuration to reload from sources."""
    global _global_config
    _global_config = None

