from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Cargar modelo Qwen2-0.5B
model_name = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_text(text, src_lang="English", tgt_lang="Spanish"):
    prompt = f"Translate from {src_lang} to {tgt_lang}: {text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ejemplos
examples = ["I like soccer", "How are you?", "What time is it?"]
for ex in examples:
    print(f"{ex} -> {translate_text(ex)}")
