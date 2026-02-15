"""
Orchestrator Node for the Therapy Workflow.
"""

from typing import Any, Dict
import structlog

from app.agents.base_agent import BaseAgent
from app.agents.state import ExamHelperState, get_conversation_context
from app.utils.intent_detector import detect_intent
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from app.tools.exam_helper_tools import get_agent_tools

logger = structlog.get_logger(__name__)


class OrchestratorNode:
    """Node for processing conversations through the orchestrator agent."""

    def __init__(self, orchestrator_agent: BaseAgent) -> None:
        self.orchestrator_agent = orchestrator_agent
        
    def process(self, state: ExamHelperState) -> Dict[str, Any]:
        """Process the current state through the orchestrator."""
        try:

            user_msg = ""
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage):
                    user_msg = msg.content
                    break

            current_intent = state.get("user_intent", "unknown")

            if current_intent == "unknown" and user_msg:
                current_intent = detect_intent(user_msg)

            tools = get_agent_tools()
            prompt = self.orchestrator_agent.get_prompt(state)

            agent = create_react_agent(
                self.orchestrator_agent.model,
                tools,
                prompt=prompt,
            )

            result = agent.invoke({"messages": state.get("messages", [])})

            return {
                "messages": result.get("messages", []),
                "user_intent": current_intent,
                "orchestrator_result": result,
            }

        except Exception as e:
            error_msg = f"Orchestrator node failed: {str(e)}"
            logger.error("Orchestrator node failed", error=str(e))
            return {
                "orchestrator_result": None,
                "error": [error_msg],
            }
