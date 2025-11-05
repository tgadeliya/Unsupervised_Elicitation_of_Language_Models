import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Literal

DATASET_SPLIT = Literal["train", "test"]


@dataclass
class TruthfulQAExample:
    question: str
    choice: str
    label: int

    @property
    def template(self):
        return dedent(f"""
        Question: {self.question}
        Claim: {self.choice}
        This claim is""")

    def template_with_label(self, label: int | None = None) -> str:
        if label is None:  
            label = self.label  # use golden label
        
        label_str = "True" if label else "False"
        return dedent(f"""
        Question: {self.question}
        Claim: {self.choice}
        This claim is {label_str}.""")

    @property
    def paper_template(self):
        return dedent(f"""
        Question: {self.question}
        Claim: {self.choice}
        This claim is [True/False] """)


class TruthfulQA:
    split_to_filename = {
        "train": "truthfulqa_train.json",
        "test": "truthfulqa_test.json",
    }

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)

    def get_dataset_split(self, split: DATASET_SPLIT) -> list[TruthfulQAExample]:
        return self._load_data(split)

    def _load_data(self, split: DATASET_SPLIT):
        data = []
        with open(self.data_dir / f"truthfulqa_{split}.json") as f:
            raw_data = json.load(f)
            for item in raw_data:
                example = TruthfulQAExample(
                    question=item["question"],
                    choice=item["choice"],
                    label=item["label"],
                )
                data.append(example)
        return data