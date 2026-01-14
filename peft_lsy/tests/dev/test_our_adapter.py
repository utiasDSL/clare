

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import OurAdapterConfig, get_peft_model, PeftModel
from peft.tuners.our_adapter.discriminator import AutoencoderConfig
from peft.tuners.our_adapter.config import FuncAdapterConfig
import torch

# ----- 1) Pick a tiny model so it runs anywhere -----
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"  # very small GPT-2 for demos

# ----- 2) Load tokenizer & base model -----
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # avoid pad issues

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="auto"  # works on CPU & GPU; set to None if you want manual .to(device)
)

# ----- 3) Define a OurAdapter config and wrap the model with PEFT -----
discriminator_cfg=AutoencoderConfig(
    feature_dim=1536
)

func_adapter_cfg=FuncAdapterConfig(
    hidden_dim=int(1536/2),
    use_lora=False
)

our_adapter_cfg = OurAdapterConfig(
    target_modules="(?P<layer_name>.+)\.(?P<layer_id>\d+)\.mlp",
    feature_dim=1536,
    discriminator_cfg=discriminator_cfg,
    func_adapter_cfg=func_adapter_cfg,
)

peft_model = get_peft_model(model, our_adapter_cfg)
peft_model.print_trainable_parameters()  # sanity check: only LoRA params should be trainable

# ----- 4) (Optional) Tiny training step just to prove it works -----
peft_model.train()
optim = torch.optim.AdamW(peft_model.parameters(), lr=2e-4)

text = "LoRA makes fine-tuning large language models efficient."
batch = tokenizer(
    [text],
    return_tensors="pt",
    padding=True,
    truncation=True
)
batch = {k: v.to(peft_model.device) for k, v in batch.items()}

# Shift labels for causal LM loss automatically via labels=batch["input_ids"]
outputs = peft_model(**batch, labels=batch["input_ids"])
loss = outputs.loss
loss.backward()
optim.step()
optim.zero_grad()
print(f"Dummy training loss: {loss.item():.4f}")

# ----- 5) Save the LoRA adapter only (small!) -----
ADAPTER_DIR = "tinygpt2-lora-adapter"
peft_model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Saved LoRA adapter to: {ADAPTER_DIR}")

# ----- 6) Load the adapter later on the same base model -----
base_model_reloaded = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float32, device_map="auto"
)
peft_model_reloaded: PeftModel = PeftModel.from_pretrained(
    base_model_reloaded, ADAPTER_DIR
).eval()

# ----- 7) (Optional) Merge LoRA weights into the base model for export/inference -----
merged_model = peft_model_reloaded.merge_and_unload()  # returns a plain HF model with weights merged
merged_model.eval()

# Quick generation demo
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)
with torch.no_grad():
    gen_ids = merged_model.generate(**inputs, max_length=50)
print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
