from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import transformers
import torch

model_base = "tiiuae/falcon-7b"
model_id = "./results/checkpoint-300"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_base, device_map={"":0})
model = PeftModel.from_pretrained(model=model, model_id=model_id, is_trainable=False, device_map={"":0})

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": 0},
)

for i in range(5):
    sequences = pipeline(
       "Sachin's first memory with a bicycle was",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

