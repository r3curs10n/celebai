import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft.tuners.lora import LoraLayer

from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")

model_id = "tiiuae/falcon-7b"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)


device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map=device_map, trust_remote_code=True
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],  # , "word_embeddings", "lm_head"],
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=10,
    learning_rate=float(2e-4),
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=10000,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

model.config.use_cache = False

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

for name, module in trainer.model.named_modules():
    # if isinstance(module, LoraLayer):
    #     if script_args.bf16:
    #         module = module.to(torch.bfloat16)
    if "norm" in name:
        module = module.to(torch.float32)
    # if "lm_head" in name or "embed_tokens" in name:
    #     if hasattr(module, "weight"):
    #         if script_args.bf16 and module.weight.dtype == torch.float32:
    #             module = module.to(torch.bfloat16)

trainer.train()
