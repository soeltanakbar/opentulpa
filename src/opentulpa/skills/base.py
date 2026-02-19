"""Base skill interface for the agent."""

from typing import Any

# Tools are async callables used by the agent runtime.
SkillTool = Any
GuidelineCondition = str
GuidelineAction = str


class Skill:
    """A skill provides tools and optional guideline (condition, action) for the agent."""

    name: str = ""
    description: str = ""

    def tools(self) -> list[SkillTool]:
        """Return tool functions to attach to the agent."""
        return []

    def guidelines(self) -> list[tuple[GuidelineCondition, GuidelineAction]]:
        """Return runtime guidance (condition, action) pairs."""
        return []

    def metadata(self) -> dict[str, Any]:
        """Optional metadata (e.g. artifact types produced)."""
        return {}
