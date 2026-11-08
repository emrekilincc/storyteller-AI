from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from deep_translator import GoogleTranslator
import torch

app = Flask(__name__)

# === Model ve Tokenizer ===
MODEL_NAME = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Çeviri Fonksiyonu ===
def translate(text, src, dest):
    try:
        return GoogleTranslator(source=src, target=dest).translate(text)
    except Exception as e:
        print(f"Çeviri hatası: {e}")
        return text

# === Hikaye Üretme Fonksiyonu ===
def generate_story(emotion, prompt, max_length=250):
    input_text = f"{emotion.lower()}: {prompt.lower()}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === Ana Sayfa ===
@app.route("/")
def home():
    return render_template("index.html")

# === API Endpoint ===
@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    emotion = data.get("emotion", "neutral")
    start_tr = data.get("start", "")

    start_en = translate(start_tr, src="tr", dest="en")
    story_en = generate_story(emotion, start_en)
    story_tr = translate(story_en, src="en", dest="tr")

    return jsonify({"story": story_tr})

# === Sunucu Başlat ===
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
