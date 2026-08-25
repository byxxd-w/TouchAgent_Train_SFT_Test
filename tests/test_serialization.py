import copy
import unittest

from touchagent_train.serialization import encode_instruct_record, validate_instruct_record

from tests.helpers import FakeTokenizer, make_record


class SerializationTest(unittest.TestCase):
    def test_variable_assistant_turns_are_fully_supervised(self):
        for assistant_turns in (2, 3, 5):
            encoded = encode_instruct_record(
                FakeTokenizer(), make_record(assistant_turns - 1), max_length=20000
            )
            self.assertEqual(len(encoded["assistant_spans"]), assistant_turns)
            supervised_positions = {
                index for index, value in enumerate(encoded["labels"]) if value != -100
            }
            span_positions = {
                index
                for start, end in encoded["assistant_spans"]
                for index in range(start, end)
            }
            self.assertEqual(supervised_positions, span_positions)
            for start, end in encoded["assistant_spans"]:
                self.assertEqual(
                    encoded["labels"][start:end], encoded["input_ids"][start:end]
                )
                self.assertEqual(encoded["input_ids"][end - 1], 151645)
                self.assertEqual(encoded["labels"][end], -100)

    def test_ordered_multi_action_turn(self):
        record = make_record()
        record["conversations"][1]["actions"] = [
            {"tool_name": "ContactFrameSelectionTool"},
            {"tool_name": "DynamicFrameSelectionTool"},
        ]
        record["conversations"][2]["value"] = (
            "ContactFrameSelectionTool output:\ncontact ready\n\n"
            "DynamicFrameSelectionTool output:\ndynamic ready"
        )
        validate_instruct_record(record)
        record["conversations"][2]["value"] = (
            "DynamicFrameSelectionTool output:\ndynamic ready\n\n"
            "ContactFrameSelectionTool output:\ncontact ready"
        )
        with self.assertRaisesRegex(ValueError, "in order"):
            validate_instruct_record(record)

    def test_invalid_contracts_are_rejected(self):
        invalid_roles = make_record()
        invalid_roles["conversations"][2]["from"] = "gpt"
        with self.assertRaisesRegex(ValueError, "alternate"):
            validate_instruct_record(invalid_roles)

        final_action = make_record()
        final_action["conversations"][-1]["actions"] = [
            {"tool_name": "AttributeAnalysisTool"}
        ]
        with self.assertRaisesRegex(ValueError, "no actions"):
            validate_instruct_record(final_action)

        legacy_schema = copy.deepcopy(make_record())
        legacy_schema["schema_version"] = "touchagent_attribute_instruct_v1"
        with self.assertRaisesRegex(ValueError, "v2"):
            validate_instruct_record(legacy_schema)


if __name__ == "__main__":
    unittest.main()
