"""Minimal model-visible protocol required by TouchAgent SFT."""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


FINAL_ANSWER_PATTERN = re.compile(r"Therefore, the answer is ([A-D])\.$")


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class AgentPhase(str, Enum):
    TOOL_SELECTION = "tool_selection"
    FINAL_ANSWER = "final_answer"


class ToolName(str, Enum):
    CONTACT_FRAME_SELECTION = "ContactFrameSelectionTool"
    DYNAMIC_FRAME_SELECTION = "DynamicFrameSelectionTool"
    ZOOM = "ZoomTool"
    ATTRIBUTE_ANALYSIS = "AttributeAnalysisTool"
    MATCHING_VERIFICATION = "MatchingVerificationTool"
    INTERACTION_KNOWLEDGE_RETRIEVAL = "InteractionKnowledgeRetrievalTool"
    SCENE_CANDIDATE_PROFILING = "SceneCandidateProfilingTool"
    SCENE_KNOWLEDGE_RETRIEVAL = "SceneKnowledgeRetrievalTool"
    RELATIVE_DEPTH_ESTIMATION = "RelativeDepthEstimationTool"
    DYNAMIC_UNDERSTANDING = "DynamicUnderstandingTool"


class AnswerLabel(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class ToolAction(StrictModel):
    tool_name: ToolName


class AgentOutput(StrictModel):
    thoughts: str = Field(min_length=1, max_length=512)
    actions: List[ToolAction] = Field(max_length=2)
    value: str = Field(min_length=1, max_length=2048)


def parse_final_answer(value: str) -> Optional[AnswerLabel]:
    match = FINAL_ANSWER_PATTERN.search(value)
    return AnswerLabel(match.group(1)) if match else None


def validate_agent_output_for_phase(
    output: AgentOutput, phase: AgentPhase
) -> Optional[AnswerLabel]:
    if phase == AgentPhase.TOOL_SELECTION:
        if not output.actions:
            raise ValueError("Tool-selection output must contain at least one action")
        if parse_final_answer(output.value) is not None:
            raise ValueError("Tool-selection output must not contain a final answer")
        return None

    if output.actions:
        raise ValueError("Final-answer output must contain no actions")
    answer = parse_final_answer(output.value)
    if answer is None:
        raise ValueError('Final value must end with "Therefore, the answer is X."')
    return answer
