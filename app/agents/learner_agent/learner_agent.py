"""
Learner Agent

Helps user in providing learning material that can be used to study a certain concept
"""

from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel

from app.agents.agent_types import LEARNER_AGENT_NAME
from app.agents.base_agent import BaseAgent
from app.agents.llm_models import LLMModels
from app.agents.state import ExamHelperState
from app.models.response_models import ExamHelperResponse

logger = structlog.get_logger(__name__)


LEARNER_AGENT_PROMPT = """You provide excellent learning material that is structured in a way that it is easy for a 
learner to study and remember any content.

YOUR ROLE:


STYLE:


CONVERSATION CONTEXT:
{context}

Remember: Keep responses under 150 words. Always end with an engaging question or invitation to share more."""


class LearnerAgent(BaseAgent):
    """Agent for handling queries related to providing easy to grasp learning material"""

    def __init__(
        self,
        agent_name: str = LEARNER_AGENT_NAME,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        model_name: str = LLMModels.GEMINI_2_5_FLASH,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            api_key=api_key,
            temperature=temperature,
            model_name=model_name,
        )

    def get_result_key(self) -> str:
        return "learner_agent_result"

    def get_prompt(self, state: Optional[ExamHelperState] = None) -> str:
        from app.agents.state import get_conversation_context

        context = get_conversation_context(state) if state else ""
        return LEARNER_AGENT_PROMPT.format(context=context)

    def get_response_format(self) -> type[BaseModel]:
        return ExamHelperResponse

    async def process_query(
        self,
        query: str,
        state: Optional[ExamHelperState] = None,
    ) -> Dict[str, Any]:
        """Process a query and provide relatded learning material"""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = self.get_prompt(state)
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=query),
            ]
            response = await self.model.ainvoke(messages)

            return {
                "success": True,
                self.get_result_key(): response.content,
                "error": [],
            }
        except Exception as e:
            logger.error("Learner agent processing failed", error=str(e))
            return {
                "success": False,
                self.get_result_key(): None,
                "error": [str(e)],
            }
