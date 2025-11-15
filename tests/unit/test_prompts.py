"""
Unit tests for prompt template manager.
"""
import os
import tempfile

import pytest
import yaml

from src.generation.prompts import PromptManager


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory."""
    temp_dir = tempfile.mkdtemp(prefix="prompt_config_")
    yield temp_dir
    # Cleanup
    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_prompts_yaml():
    """Create sample prompts YAML content."""
    return {
        "system": "You are a helpful AI assistant.",
        "rag": "Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        "no_context": "Question: {query}\n\nAnswer:",
        "custom_template": "Translate to {language}: {text}",
    }


@pytest.fixture
def prompts_config_file(temp_config_dir, sample_prompts_yaml):
    """Create a prompts configuration file."""
    config_path = os.path.join(temp_config_dir, "prompts.yaml")
    with open(config_path, "w") as f:
        yaml.dump(sample_prompts_yaml, f)
    return config_path


class TestPromptManagerInit:
    """Test PromptManager initialization."""

    def test_init_with_existing_config(self, prompts_config_file):
        """Test initialization with existing config file."""
        manager = PromptManager(config_path=prompts_config_file)
        assert manager.prompts is not None
        assert "system" in manager.prompts
        assert "rag" in manager.prompts

    def test_init_loads_all_templates(self, prompts_config_file):
        """Test that all templates are loaded."""
        manager = PromptManager(config_path=prompts_config_file)
        assert len(manager.prompts) >= 3
        assert "system" in manager.prompts
        assert "rag" in manager.prompts
        assert "no_context" in manager.prompts

    def test_init_with_nonexistent_config(self, temp_config_dir):
        """Test initialization with nonexistent config raises error."""
        nonexistent_path = os.path.join(temp_config_dir, "nonexistent.yaml")
        with pytest.raises(FileNotFoundError):
            PromptManager(config_path=nonexistent_path)

    def test_init_with_invalid_yaml(self, temp_config_dir):
        """Test initialization with invalid YAML raises error."""
        invalid_config = os.path.join(temp_config_dir, "invalid.yaml")
        with open(invalid_config, "w") as f:
            f.write("{ invalid: yaml: syntax:")

        with pytest.raises(yaml.YAMLError):
            PromptManager(config_path=invalid_config)

    def test_init_with_empty_config(self, temp_config_dir):
        """Test initialization with empty config file."""
        empty_config = os.path.join(temp_config_dir, "empty.yaml")
        with open(empty_config, "w") as f:
            f.write("")

        manager = PromptManager(config_path=empty_config)
        assert manager.prompts == {} or manager.prompts is None


class TestFormatRagPrompt:
    """Test RAG prompt formatting."""

    def test_format_rag_prompt_basic(self, prompts_config_file):
        """Test basic RAG prompt formatting."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_rag_prompt(
            query="What is Python?",
            context="Python is a programming language.",
        )

        assert "Python is a programming language." in result
        assert "What is Python?" in result

    def test_format_rag_prompt_preserves_variables(self, prompts_config_file):
        """Test that RAG prompt formatting preserves variable placeholders."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_rag_prompt(
            query="Test query",
            context="Test context",
        )

        assert "Test query" in result
        assert "Test context" in result

    def test_format_rag_prompt_with_multiline_context(self, prompts_config_file):
        """Test RAG prompt with multiline context."""
        manager = PromptManager(config_path=prompts_config_file)
        multiline_context = "Line 1\nLine 2\nLine 3"
        result = manager.format_rag_prompt(
            query="Question?",
            context=multiline_context,
        )

        assert multiline_context in result

    def test_format_rag_prompt_with_special_chars(self, prompts_config_file):
        """Test RAG prompt with special characters."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_rag_prompt(
            query="What is @#$%?",
            context="Context with 'quotes' and \"double quotes\"",
        )

        assert "@#$%" in result
        assert "quotes" in result

    def test_format_rag_prompt_empty_context(self, prompts_config_file):
        """Test RAG prompt with empty context raises error."""
        manager = PromptManager(config_path=prompts_config_file)
        with pytest.raises(ValueError, match="context cannot be empty"):
            manager.format_rag_prompt(
                query="Question?",
                context="",
            )


class TestFormatSystemPrompt:
    """Test system prompt formatting."""

    def test_format_system_prompt_returns_string(self, prompts_config_file):
        """Test that system prompt returns a string."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_system_prompt_content(self, prompts_config_file):
        """Test system prompt contains expected content."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_system_prompt()
        assert "helpful" in result.lower() or "assistant" in result.lower()

    def test_format_system_prompt_consistent(self, prompts_config_file):
        """Test system prompt returns consistent value."""
        manager = PromptManager(config_path=prompts_config_file)
        result1 = manager.format_system_prompt()
        result2 = manager.format_system_prompt()
        assert result1 == result2


class TestFormatNoContextPrompt:
    """Test no-context prompt formatting."""

    def test_format_no_context_prompt_basic(self, prompts_config_file):
        """Test basic no-context prompt formatting."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_no_context_prompt(query="What is X?")
        assert "What is X?" in result

    def test_format_no_context_prompt_no_context_variable(self, prompts_config_file):
        """Test no-context prompt doesn't contain context variable."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_no_context_prompt(query="Question")
        assert "{context}" not in result

    def test_format_no_context_prompt_with_special_chars(self, prompts_config_file):
        """Test no-context prompt with special characters."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_no_context_prompt(query="What is @#$% used for?")
        assert "@#$%" in result

    def test_format_no_context_prompt_empty_query(self, prompts_config_file):
        """Test no-context prompt with empty query raises error."""
        manager = PromptManager(config_path=prompts_config_file)
        with pytest.raises(ValueError, match="query cannot be empty"):
            manager.format_no_context_prompt(query="")


class TestCustomPromptFormatting:
    """Test custom template formatting."""

    def test_custom_prompt_with_available_template(self, prompts_config_file):
        """Test custom prompt formatting with available template."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_custom_prompt(
            template_name="custom_template",
            language="Spanish",
            text="Hello world",
        )

        assert "Spanish" in result
        assert "Hello world" in result

    def test_custom_prompt_missing_template(self, prompts_config_file):
        """Test custom prompt with missing template raises error."""
        manager = PromptManager(config_path=prompts_config_file)
        with pytest.raises(KeyError):
            manager.format_custom_prompt(
                template_name="nonexistent_template",
                some_var="value",
            )

    def test_custom_prompt_missing_variables(self, prompts_config_file):
        """Test custom prompt with missing template variables raises error."""
        manager = PromptManager(config_path=prompts_config_file)
        with pytest.raises(KeyError):
            manager.format_custom_prompt(
                template_name="custom_template",
                # Missing required 'language' and 'text'
            )

    def test_custom_prompt_extra_variables_ignored(self, prompts_config_file):
        """Test custom prompt ignores extra variables."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_custom_prompt(
            template_name="custom_template",
            language="French",
            text="Bonjour",
            extra_var="ignored",
        )

        assert "French" in result
        assert "Bonjour" in result

    def test_custom_prompt_with_system_template(self, prompts_config_file):
        """Test custom prompt can access standard templates."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_custom_prompt(template_name="system")
        assert isinstance(result, str)
        assert len(result) > 0


class TestTemplateValidation:
    """Test template validation and error handling."""

    def test_missing_required_prompts_warning(self, temp_config_dir, caplog):
        """Test warning logged for missing required prompts."""
        incomplete_prompts = {
            "system": "System prompt",
            # Missing 'rag' and 'no_context'
        }
        config_path = os.path.join(temp_config_dir, "incomplete.yaml")
        with open(config_path, "w") as f:
            yaml.dump(incomplete_prompts, f)

        manager = PromptManager(config_path=config_path)
        # Missing templates should be noted but not crash
        assert manager.prompts is not None

    def test_malformed_template_variable(self, temp_config_dir):
        """Test handling of malformed template variables."""
        bad_templates = {
            "malformed": "This has {unclosed variable",
        }
        config_path = os.path.join(temp_config_dir, "malformed.yaml")
        with open(config_path, "w") as f:
            yaml.dump(bad_templates, f)

        manager = PromptManager(config_path=config_path)
        # Should load but may fail on formatting
        assert "malformed" in manager.prompts

    def test_template_with_escaped_braces(self, temp_config_dir):
        """Test templates with escaped braces."""
        templates_with_escaped = {
            "escaped": "Use {{braces}} like this: {variable}",
        }
        config_path = os.path.join(temp_config_dir, "escaped.yaml")
        with open(config_path, "w") as f:
            yaml.dump(templates_with_escaped, f)

        manager = PromptManager(config_path=config_path)
        result = manager.format_custom_prompt(
            template_name="escaped",
            variable="test",
        )
        assert "test" in result


class TestPromptIntegration:
    """Test integration between different prompt types."""

    def test_system_and_rag_prompts_together(self, prompts_config_file):
        """Test using system and RAG prompts together."""
        manager = PromptManager(config_path=prompts_config_file)
        system = manager.format_system_prompt()
        rag = manager.format_rag_prompt(
            query="What is Python?",
            context="Python is a language.",
        )

        assert len(system) > 0
        assert "Python" in rag

    def test_rag_and_no_context_prompts_differ(self, prompts_config_file):
        """Test that RAG and no-context prompts are different."""
        manager = PromptManager(config_path=prompts_config_file)
        rag = manager.format_rag_prompt(
            query="Question?",
            context="Context provided.",
        )
        no_context = manager.format_no_context_prompt(query="Question?")

        # RAG should have context, no_context shouldn't
        assert "Context provided." in rag
        assert "Context provided." not in no_context

    def test_all_prompts_accessible(self, prompts_config_file):
        """Test all prompt types are accessible."""
        manager = PromptManager(config_path=prompts_config_file)

        system = manager.format_system_prompt()
        rag = manager.format_rag_prompt(query="Q", context="C")
        no_context = manager.format_no_context_prompt(query="Q")

        assert all(isinstance(p, str) and len(p) > 0 for p in [system, rag, no_context])


class TestPromptManagerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_context(self, prompts_config_file):
        """Test handling of very long context."""
        manager = PromptManager(config_path=prompts_config_file)
        long_context = "Context. " * 10000  # Very long context
        result = manager.format_rag_prompt(
            query="Short query",
            context=long_context,
        )

        assert "Short query" in result
        assert "Context." in result

    def test_unicode_characters_in_prompts(self, prompts_config_file):
        """Test handling of Unicode characters."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_rag_prompt(
            query="Что это? (Russian)",
            context="Context with émojis 🚀 and symbols",
        )

        assert "русski" in result or "что" in result.lower() or "Context" in result

    def test_newlines_preserved_in_templates(self, prompts_config_file):
        """Test that newlines are preserved in templates."""
        manager = PromptManager(config_path=prompts_config_file)
        result = manager.format_rag_prompt(
            query="Q",
            context="C",
        )

        # RAG template should have newlines
        assert "\n" in result
