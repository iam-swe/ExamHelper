"""
Synthesizer Node for the Therapy Workflow.
"""

from typing import Any, Dict

import structlog
from langchain_core.messages import AIMessage

from app.agents.state import ExamHelperState
from app.agents.synthesizer_agent.synthesizer_agent import SynthesizerAgent

logger = structlog.get_logger(__name__)


class SynthesizerNode:
    """Node for synthesizing and polishing final responses."""

    def __init__(self, synthesizer_agent: SynthesizerAgent) -> None:
        self.synthesizer_agent = synthesizer_agent

    def process(self, state: ExamHelperState) -> Dict[str, Any]:
        """Process and polish the current response."""
        try:
            polished_response = state.get("orchestrator_result", "")

            new_messages = list(state.get("messages", []))
            for i in range(len(new_messages) - 1, -1, -1):
                if isinstance(new_messages[i], AIMessage) and not getattr(new_messages[i], "tool_calls", None):
                    new_messages[i] = AIMessage(content=polished_response)
                    break

            new_turn_count = state.get("turn_count", 0) + 1

            return {
                "messages": new_messages,
                "current_response": polished_response,
                "turn_count": new_turn_count,
            }

        except Exception as e:
            error_msg = f"Synthesizer node failed: {str(e)}"
            logger.error("Synthesizer node failed", error=str(e))
            return {
                "error": [error_msg],
            }
