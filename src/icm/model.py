import math
import os
from abc import abstractmethod

import numpy as np
import openai
from dotenv import load_dotenv
from tqdm import tqdm

from icm.dataset import TruthfulQAExample

load_dotenv()
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")


class Model:
    def __init__(self, model_name: str, system_prompt: str = ""):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.client = openai.OpenAI(
            api_key=HYPERBOLIC_API_KEY,
            base_url="https://api.hyperbolic.xyz/v1",
        )

    def predict(
        self,
        dataset: list[TruthfulQAExample],
        temperature: float = 0.0,
    ) -> list[int]:
        preds = []
        for example in tqdm(
            dataset, desc="Running prediction for model: " + self.model_name
        ):
            response = self.generate(example=example, temperature=temperature)
            pred = self.response_to_label(response)
            preds.append(pred)
        return preds

    @abstractmethod
    def response_to_label(self, response) -> int:
        pass

    @abstractmethod
    def generate(self, example: TruthfulQAExample, temperature: float = 0.0):
        pass


class BaseModel(Model):
    def generate(self, example: TruthfulQAExample, temperature: float = 0.0):
        prompt = self.system_prompt + "\n" + example.template
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=1,
            logprobs=20,
        )
        return response

    def response_to_label(self, response) -> int:
        logprobs = response.choices[0].logprobs.top_logprobs[0]
        diff = self.extract_diff_true_false_probs(logprobs)
        if diff is None:
            return -1
        return 1 if diff > 0 else 0

    def extract_diff_true_false_probs(
        self, top_logprobs: dict[str, float], eps=1e-5
    ) -> float | None:
        """Logic adapted from the official repository"""
        probs = {"false": eps, "true": eps}
        for token, logprob in top_logprobs.items():
            if "false" in token.lower():
                probs["false"] += math.exp(logprob)
            elif "true" in token.lower():
                probs["true"] += math.exp(logprob)

        if probs["false"] == eps and probs["true"] == eps:
            return None
        else:
            return math.log(probs["true"]) - math.log(probs["false"])


class ChatModel(Model):
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        use_paper_template: bool = False,
        use_system_prompt: bool = False,
    ):
        super().__init__(model_name, system_prompt)
        self.use_paper_template = use_paper_template
        self.use_system_prompt = use_system_prompt

    def generate(
        self,
        example: TruthfulQAExample,
        temperature: float = 0,
    ):
        if self.use_paper_template:
            prompt = example.paper_template
        else:
            prompt = example.template
        messages: list[dict] = [
            # paper does not mention system prompt or special instruction
            {"role": "user", "content": prompt},
        ]
        if self.use_system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1,
        )
        return response

    def response_to_label(self, response) -> int:
        completion = response.choices[0].message.content.lower()
        if "true" in completion:
            return 1
        elif "false" in completion:
            return 0
        else:
            return -1  # neither true or false predicted


class ICMModel:
    def __init__(self, model_name: str, random_seed: int = 25):
        self.model_name = model_name
        self.system_prompt = None
        self.client = openai.OpenAI(
            api_key=HYPERBOLIC_API_KEY,
            base_url="https://api.hyperbolic.xyz/v1",
        )
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def predict(
        self,
        dataset: list[TruthfulQAExample],
        init_temperature: float = 10.0,
        final_temperature: float = 0.01,
        cooling_rate: float = 0.99,
        K: int = 8,
    ) -> list[int]:
        N = len(dataset)
        labels = self._init_labels(N, K)

        # Add weights for convergence
        # Without logical consistency
        idxs_to_label = self.rng.choice(N, N, replace=False)
        for step, idx_to_label in tqdm(enumerate(idxs_to_label, start=1), "ICM search"):
            # Followed implementation from the paper Algo 1,
            # despite in the repo it is different
            T = max(
                final_temperature,
                init_temperature / (1 + cooling_rate * math.log(step)),
            )

            labels_hat = labels.copy()
            labels_hat[idx_to_label] = self.calculate_predictability_single(  # type: ignore
                idx_to_label, dataset, labels_hat
            )["label"]
            U = self.calculate_mutual_predictability(dataset, labels)
            U_hat = self.calculate_mutual_predictability(dataset, labels_hat)
            delta = U_hat - U
            if delta >= 0 or self.rng.uniform(0, 1) < math.exp(delta / T):
                labels = labels_hat
            print(f"Labels after {step=}: {labels}")
        return labels  # type: ignore

    def calculate_predictability_single(
        self,
        idx: int,
        dataset: list[TruthfulQAExample],
        labeled_dataset: list[int | None],
    ) -> dict[str, float | dict[int, float] | None]:
        # Due to limited time assumed prompt caching
        # is working and no additional implementation
        # is needed
        prompt = self._compose_prompt(idx, dataset, labeled_dataset)
        logprobs = self._generate_logprobs(prompt)
        logprobs_per_label = self._get_logprobs_per_label(logprobs)
        label = 1 if logprobs_per_label[1] > logprobs_per_label[0] else 0
        return {
            "label": label,
            "label_logprob": logprobs_per_label[label],
            "logprobs_per_label": logprobs_per_label,
        }

    def calculate_mutual_predictability(
        self, dataset: list[TruthfulQAExample], labels: list[int | None]
    ) -> float:
        mp_score = 0.0
        for idx in range(len(labels)):
            if labels[idx] is None or labels[idx] == -1:
                continue
            pred = self.calculate_predictability_single(idx, dataset, labels)
            label_logprob = pred["logprobs_per_label"][labels[idx]]  # type: ignore
            mp_score += label_logprob
        # ignore alpha
        return mp_score

    def _init_labels(self, N: int, K: int) -> list[int | None]:
        "Create empty label set with K randomly selected and labeled examples."
        labels = np.array([None] * N)
        # labels initialization used from
        # the original implementation
        init_labels = [1] * (K // 2) + [0] * (K // 2)
        initialization_idxs = self.rng.choice(N, K, replace=False)
        labels[initialization_idxs] = init_labels
        return labels.tolist()

    def _compose_prompt(
        self, idx: int, dataset: list[TruthfulQAExample], labels: list[int | None]
    ) -> str:
        context = []
        for lidx, label in enumerate(labels):
            # Use only correctly (true or False) labeled
            # examples for the context
            if label in [0, 1] and lidx != idx:
                # and example with currently assigned label to the context
                context.append(dataset[lidx].template_with_label(label))
        context.append(dataset[idx].template)
        return "\n".join(context)

    def _generate_logprobs(self, prompt: str) -> dict[str, float]:
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=0,
            max_tokens=1,
            logprobs=20,
        )
        return response.choices[0].logprobs.top_logprobs[0]  # type: ignore

    def _get_logprobs_per_label(
        self, logprobs: dict[str, float], eps=1e-5
    ) -> None | dict[int, float]:
        """Get logprobs for true and false label.
        Logic and eps default value adapted from the official implementation"""
        probs = {0: eps, 1: eps}
        for token, logprob in logprobs.items():
            if "false" in token.lower():
                probs[0] += math.exp(logprob)
            elif "true" in token.lower():
                probs[1] += math.exp(logprob)

        probs = {l: math.log(p) for l, p in probs.items()}
        return probs
