import importlib
from pathlib import Path
from typing import Any

from github_intelligence.post_training.dataset import load_policy_dataset
from github_intelligence.post_training.formatting import (
    POLICY_SYSTEM_PROMPT,
    completion_reward,
    policy_prompt,
    sft_record,
)
from github_intelligence.post_training.models import DatasetSplit, PolicyExample


def require_rl_stack() -> dict[str, Any]:
    """Load heavyweight, optional training libraries only for an explicit training run."""

    modules: dict[str, Any] = {}
    missing: list[str] = []
    for name in ("datasets", "peft", "torch", "transformers", "trl"):
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing RL dependencies ({names}). Install with: pip install -e '.[rl]'"
        )
    return modules


def train_sft(
    dataset_path: Path,
    output_dir: Path,
    *,
    model_name: str,
    max_steps: int,
    seed: int,
) -> None:
    stack = require_rl_stack()
    records = [sft_record(item) for item in _training_examples(dataset_path)]
    dataset = stack["datasets"].Dataset.from_list(records)
    peft_config = _lora_config(stack["peft"])
    config = stack["trl"].SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        assistant_only_loss=True,
        bf16=True,
        logging_steps=5,
        save_steps=max(25, max_steps // 2),
        seed=seed,
        report_to="none",
        model_init_kwargs=_model_init_kwargs(stack["transformers"], stack["torch"]),
    )
    trainer = stack["trl"].SFTTrainer(
        model=model_name,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))


def train_grpo(
    dataset_path: Path,
    output_dir: Path,
    *,
    model_name: str,
    max_steps: int,
    seed: int,
    num_generations: int = 4,
) -> None:
    stack = require_rl_stack()
    examples = _training_examples(dataset_path)
    by_id = {item.example_id: item for item in examples}
    records = [
        {
            "prompt": [
                {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": policy_prompt(item)},
            ],
            "example_id": item.example_id,
        }
        for item in examples
    ]
    dataset = stack["datasets"].Dataset.from_list(records)

    def reward_func(
        completions: list[Any], example_id: list[str], **_: Any
    ) -> list[float]:
        rewards: list[float] = []
        for completion, identifier in zip(completions, example_id, strict=True):
            text = _completion_text(completion)
            rewards.append(completion_reward(by_id[identifier], text))
        return rewards

    model = _load_trainable_adapter(stack, model_name)
    config = stack["trl"].GRPOConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=5e-7,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        num_generations=num_generations,
        max_completion_length=192,
        temperature=0.8,
        logging_steps=5,
        save_steps=max(25, max_steps // 2),
        seed=seed,
        report_to="none",
        log_completions=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    trainer = stack["trl"].GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir))


def _training_examples(path: Path) -> list[PolicyExample]:
    examples = [item for item in load_policy_dataset(path) if item.split == DatasetSplit.TRAIN]
    if not examples:
        raise ValueError("Policy dataset contains no training examples")
    return examples


def _lora_config(peft: Any) -> Any:
    return peft.LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def _model_init_kwargs(transformers: Any, torch: Any) -> dict[str, Any]:
    return {
        "quantization_config": transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        "torch_dtype": torch.bfloat16,
    }


def _load_trainable_adapter(stack: dict[str, Any], adapter_path: str) -> Any:
    path = Path(adapter_path)
    if not (path / "adapter_config.json").exists():
        raise ValueError("GRPO must start from the saved SFT adapter directory")
    peft_config = stack["peft"].PeftConfig.from_pretrained(adapter_path)
    base = stack["transformers"].AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        **_model_init_kwargs(stack["transformers"], stack["torch"]),
    )
    base = stack["peft"].prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True
    )
    return stack["peft"].PeftModel.from_pretrained(base, adapter_path, is_trainable=True)


def load_inference_model(model_name: str) -> tuple[Any, Any]:
    stack = require_rl_stack()
    path = Path(model_name)
    if (path / "adapter_config.json").exists():
        peft_config = stack["peft"].PeftConfig.from_pretrained(model_name)
        base_name = str(peft_config.base_model_name_or_path)
        base = stack["transformers"].AutoModelForCausalLM.from_pretrained(
            base_name,
            **_model_init_kwargs(stack["transformers"], stack["torch"]),
        )
        model = stack["peft"].PeftModel.from_pretrained(base, model_name)
        tokenizer_name = (
            model_name if (path / "tokenizer_config.json").exists() else base_name
        )
    else:
        model = stack["transformers"].AutoModelForCausalLM.from_pretrained(
            model_name,
            **_model_init_kwargs(stack["transformers"], stack["torch"]),
        )
        tokenizer_name = model_name
    tokenizer = stack["transformers"].AutoTokenizer.from_pretrained(tokenizer_name)
    model.eval()
    return model, tokenizer


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)
