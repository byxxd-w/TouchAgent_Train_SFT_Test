from pathlib import Path

from touchagent_train.config import SFTConfig


TOOLS = [
    "ContactFrameSelectionTool",
    "DynamicFrameSelectionTool",
    "ZoomTool",
    "AttributeAnalysisTool",
    "MatchingVerificationTool",
    "InteractionKnowledgeRetrievalTool",
    "SceneCandidateProfilingTool",
    "SceneKnowledgeRetrievalTool",
    "RelativeDepthEstimationTool",
    "DynamicUnderstandingTool",
]


class FakeTokenizer:
    pad_token_id = 0

    @staticmethod
    def _content_ids(value):
        return [1000 + ord(char) for char in value]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if not tokenize:
            raise AssertionError("Tests require tokenized chat templates")
        role_ids = {"system": 10, "user": 11, "assistant": 12}
        values = []
        for message in messages:
            values.append(role_ids[message["role"]])
            values.extend(self._content_ids(message["content"]))
            values.extend((151645, 198))
        if add_generation_prompt:
            values.append(role_ids["assistant"])
        return values

    def convert_tokens_to_ids(self, token):
        return 151645 if token == "<|im_end|>" else -1


def make_record(tool_count=1, record_id="episode_1"):
    conversations = [{"from": "human", "value": "Question with A/B/C/D."}]
    for index in range(tool_count):
        tool = TOOLS[index % len(TOOLS)]
        conversations.extend(
            (
                {
                    "from": "gpt",
                    "thoughts": f"tool thought {index}",
                    "actions": [{"tool_name": tool}],
                    "value": f"call {tool}",
                },
                {"from": "human", "value": f"{tool} output:\nobservation {index}"},
            )
        )
    conversations.append(
        {
            "from": "gpt",
            "thoughts": "final thought",
            "actions": [],
            "value": "Evidence supports option C. Therefore, the answer is C.",
        }
    )
    return {
        "schema_version": "touchagent_attribute_instruct_v2",
        "id": record_id,
        "conversations": conversations,
    }


def make_config(root: Path, data_path: Path, manifest_path: Path) -> SFTConfig:
    return SFTConfig.model_validate(
        {
            "schema_version": "touchagent.sft_config.v2",
            "dataset_version": "TouchAgent-Instruct-v2",
            "model_name_or_path": str((root / "model").resolve()),
            "output_dir": str((root / "output").resolve()),
            "data_path": str(data_path.resolve()),
            "data_manifest_path": str(manifest_path.resolve()),
            "num_train_epochs": 5.0,
            "max_steps": 3,
            "save_steps": 3,
            "save_total_limit": 1,
            "deepspeed_config_path": str((root / "zero2.json").resolve()),
        }
    )
