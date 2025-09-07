"""
model_utils.py — Интеграция LoRA и QAT для Daur-AI
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# LoRA: загрузка адаптера и применение к LLM

def load_lora_model(base_model_name: str, lora_path: str):
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    peft_config = PeftConfig.from_pretrained(lora_path)
    lora_model = PeftModel.from_pretrained(model, lora_path, config=peft_config)
    return lora_model, tokenizer

# QAT: пример настройки для PyTorch

def prepare_qat_model(model):
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    return model

# После обучения:
# torch.quantization.convert(model.eval(), inplace=True)
