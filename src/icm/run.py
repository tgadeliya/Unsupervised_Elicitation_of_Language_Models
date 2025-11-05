from importlib.resources import files
from pathlib import Path

import numpy as np

from icm.dataset import TruthfulQA
from icm.metrics import accuracy
from icm.model import BaseModel, ChatModel, ICMModel
from icm.plot import plot_truthfulqa_single

LLAMA_BASE = "meta-llama/Meta-Llama-3.1-405B"
LLAMA_CHAT = "meta-llama/Meta-Llama-3.1-405B-Instruct"


def replicate_figure_1_truthfulqa():
    accuracy_dict = {}

    path = str(Path(__file__).parent.parent.parent / "data")
    dataset = TruthfulQA(data_dir=path)
    test_dataset = dataset.get_dataset_split("test")
    true_labels = [ex.label for ex in test_dataset]

    # Zero-shot
    hhh_prompt = (files("icm.prompts") / "system_prompt").read_text("utf-8")
    base_model = BaseModel(model_name=LLAMA_BASE, system_prompt=hhh_prompt)
    preds_base = base_model.predict(test_dataset)
    accuracy_dict["Zero-shot"] = [accuracy(preds=preds_base, true_labels=true_labels)]

    # Zero-shot (chat)
    chat_system_prompt = (
        "You should evaluate whether following claim True or False. Generate only one token 'True' or 'False'."
    )
    chat_model = ChatModel(
        model_name=LLAMA_CHAT,
        system_prompt=chat_system_prompt,
        use_system_prompt=True,
    )
    preds_chat = chat_model.predict(test_dataset)
    accuracy_dict["Zero-shot (chat)"] = [accuracy(preds=preds_chat, true_labels=true_labels)]

    random_seeds = [25, 7, 14]
    # Golden Supervision (many-shot)
    many_shot_accuracy_rs = []

    for random_seed in random_seeds:
        train_dataset = dataset.get_dataset_split("train")
        np.random.default_rng(random_seed).shuffle(train_dataset)
        many_shot_prompt = "\n".join([ex.template_with_label() for ex in train_dataset])
        many_shot_base_model = BaseModel(
            model_name=LLAMA_BASE,
            system_prompt=many_shot_prompt,
        )
        preds_many_shot = many_shot_base_model.predict(test_dataset)
        many_shot_accuracy_rs.append(accuracy(preds=preds_many_shot, true_labels=true_labels))
    accuracy_dict["Golden supervision"] = many_shot_accuracy_rs

    # ICM method
    icm_accuracy_rs = []
    for random_seed in random_seeds:
        icm_model = ICMModel(model_name=LLAMA_BASE, random_seed=random_seed)
        icm_pred = icm_model.predict(
            dataset=test_dataset,
            init_temperature=1.0,
            final_temperature=0.1,
            cooling_rate=0.01,
            K=8,
        )
        icm_accuracy_rs.append(accuracy(preds=icm_pred, true_labels=true_labels))
    accuracy_dict["ICM"] = icm_accuracy_rs

    print(accuracy_dict)
    plot_truthfulqa_single(accuracy_dict, "truthfulqa_2.png")


def main():
    replicate_figure_1_truthfulqa()


if __name__ == "__main__":
    main()
