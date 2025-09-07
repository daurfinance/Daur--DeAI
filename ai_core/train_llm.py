"""
train_llm.py — Пример обучения LLM для Daur-AI
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig
import torch

# 1. Загрузка датасета
# Датасеты должны быть в формате .txt (один пример на строку)
train_path = "ai_core/datasets/train.txt"
test_path = "ai_core/datasets/test.txt"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128
)
test_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=test_path,
    block_size=128
)

# 2. LoRA/QAT конфиг
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, lora_config)
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# 3. Аргументы обучения
training_args = TrainingArguments(
    output_dir="ai_core/models/llama-lora-qat",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100
)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. Запуск обучения
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator
)
trainer.train()

# 5. Сохранение модели для инференса (ONNX/GGML)
model.save_pretrained("ai_core/models/llama-lora-qat")
# Для экспорта в ONNX:
# torch.onnx.export(model, torch.randn(1, 128), "ai_core/models/llama-lora-qat/model.onnx")
