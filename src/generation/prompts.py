"""
Prompt template management for RAG pipeline.
"""
import logging
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates for LLM generation.
    
    Loads and formats prompt templates from YAML configuration,
    supporting system prompts, RAG prompts, and no-context fallbacks.
    """

    def __init__(self, config_path: str = "config/prompts.yaml"):
        """
        Initialize prompt manager by loading templates from YAML.

        Args:
            config_path: Path to prompts.yaml configuration file

        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails
            KeyError: If required prompts not in config
        """
        self.config_path = config_path
        self.prompts = {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                self.prompts = config or {}

            logger.info(f"Loaded prompts from {config_path}")

            # Validate required prompts exist
            required = ["system", "rag", "no_context"]
            missing = [p for p in required if p not in self.prompts]
            if missing:
                logger.warning(f"Missing prompt templates in config: {missing}")

        except FileNotFoundError:
            logger.error(f"Prompts config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse prompts YAML: {e}")
            raise

    def format_system_prompt(self) -> str:
        """
        Get system prompt for setting LLM behavior.

        Returns:
            System prompt string (empty if not configured)
        """
        prompt = self.prompts.get("system", "")
        if not prompt:
            logger.warning("System prompt not configured")
        return prompt

    def format_rag_prompt(self, query: str, context: str) -> str:
        """
        Format RAG prompt with context and query.

        Args:
            query: User question
            context: Retrieved context chunks

        Returns:
            Formatted prompt with context and query

        Raises:
            ValueError: If query or context is empty
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if not context or not context.strip():
            raise ValueError("context cannot be empty")

        template = self.prompts.get("rag", "")
        if not template:
            # Fallback to simple format if not in config
            logger.warning("RAG prompt not configured, using default format")
            template = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            formatted = template.format(context=context, query=query)
            logger.debug(f"Formatted RAG prompt (context_len: {len(context)}, query_len: {len(query)})")
            return formatted
        except KeyError as e:
            logger.error(f"RAG prompt template missing required field: {e}")
            raise

    def format_no_context_prompt(self, query: str) -> str:
        """
        Format prompt when no context is available.

        Args:
            query: User question

        Returns:
            Formatted prompt for question-only generation

        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        template = self.prompts.get("no_context", "")
        if not template:
            # Fallback to simple format if not in config
            logger.warning("No-context prompt not configured, using default format")
            template = "Question: {query}\n\nAnswer:"

        try:
            formatted = template.format(query=query)
            logger.debug(f"Formatted no-context prompt (query_len: {len(query)})")
            return formatted
        except KeyError as e:
            logger.error(f"No-context prompt template missing required field: {e}")
            raise

    def format_custom_prompt(self, template_name: str, **kwargs) -> str:
        """
        Format custom prompt template with provided values.

        Args:
            template_name: Name of template in config
            **kwargs: Values to substitute in template

        Returns:
            Formatted prompt

        Raises:
            KeyError: If template or required fields not found
        """
        template = self.prompts.get(template_name)
        if not template:
            raise KeyError(f"Prompt template '{template_name}' not found in config")

        try:
            formatted = template.format(**kwargs)
            logger.debug(f"Formatted custom prompt '{template_name}'")
            return formatted
        except KeyError as e:
            logger.error(f"Custom prompt template missing required field: {e}")
            raise

    def get_prompt(self, name: str) -> Optional[str]:
        """
        Get raw prompt template by name.

        Args:
            name: Template name in config

        Returns:
            Template string or None if not found
        """
        return self.prompts.get(name)

    def list_prompts(self) -> list:
        """
        List all available prompt templates.

        Returns:
            List of prompt template names
        """
        return list(self.prompts.keys())
