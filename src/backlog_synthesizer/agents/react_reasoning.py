"""ReAct (Reason + Act) reasoning module for the Orchestrator Agent.

Provides LLM-based decision-making at pipeline decision points.
The Orchestrator calls this module when trigger conditions are met
(e.g., empty parser output, all duplicates, quality issues).

The ReAct layer is purely advisory — if the LLM call fails or returns
an unparseable response, the first available action is used as default.
"""

import asyncio
import json
import logging

from backlog_synthesizer.tools.interfaces import LLMGenerationTool

logger = logging.getLogger(__name__)


class ReActReasoner:
    """Uses LLM to make decisions at pipeline decision points."""

    def __init__(self, llm_tool: LLMGenerationTool | None = None) -> None:
        self._llm_tool = llm_tool

    async def decide(
        self,
        decision_point: str,
        observation: str,
        available_actions: list[dict],
    ) -> dict:
        """Ask LLM to decide what to do.

        Args:
            decision_point: Name of the decision point (e.g., "after_parser_empty").
            observation: What was observed that triggered this decision.
            available_actions: List of {"action": name, "description": text}.

        Returns:
            {"action": chosen_action, "reason": explanation, "parameters": {}}
            Falls back to the first action if LLM call fails.
        """
        default_action = {
            "action": available_actions[0]["action"],
            "reason": "default fallback",
            "parameters": {},
        }

        if self._llm_tool is None:
            return default_action

        # Build prompt
        actions_text = "\n".join(
            f"  - {a['action']}: {a['description']}" for a in available_actions
        )
        prompt = (
            f"You are an orchestrator reasoning about a pipeline decision point.\n\n"
            f"Decision Point: {decision_point}\n"
            f"Observation: {observation}\n\n"
            f"Available Actions:\n{actions_text}\n\n"
            f"Choose the best action. Respond with ONLY a JSON object:\n"
            f'{{"action": "<chosen_action>", "reason": "<brief explanation>", "parameters": {{}}}}\n'
        )

        system_prompt = (
            "You are a pipeline orchestrator making routing decisions. "
            "Respond with only valid JSON, no markdown fences or extra text."
        )

        try:
            response = await asyncio.to_thread(
                self._llm_tool.generate, prompt, system_prompt
            )

            # Strip markdown code fences if present
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                # Remove first line (```json) and last line (```)
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response = "\n".join(lines)

            parsed = json.loads(response)

            # Validate the action is one of the available ones
            valid_actions = {a["action"] for a in available_actions}
            if parsed.get("action") not in valid_actions:
                logger.warning(
                    "ReAct LLM returned invalid action '%s', falling back to default",
                    parsed.get("action"),
                )
                return default_action

            return {
                "action": parsed["action"],
                "reason": parsed.get("reason", "no reason provided"),
                "parameters": parsed.get("parameters", {}),
            }

        except Exception as e:
            logger.warning(
                "ReAct reasoning failed at '%s': %s. Using default action.",
                decision_point,
                e,
            )
            return default_action
