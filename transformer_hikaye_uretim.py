#bu kodlarda kullanıcıdan türkçe giriş alıyor. bunu ingilizceye çevirip buna uygun hikaye üretiyor. daha sonra bu metni metni tekrar türkçye çevirip kullanıcıya sunuyor.
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from deep_translator import GoogleTranslator


# === Model ve Tokenizer'ı yükle ===
MODEL_NAME = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # cuda mı yoksa cpu mu kullanılacağına karar verilir.
model.to(device)

from deep_translator import GoogleTranslator

def translate(text, src_lang, dest_lang):
    try:
        return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    except Exception as e:
        print(f"Çeviri hatası: {e}")
        return text


def generate_story(emotion, start_text, max_length=150, temperature=0.8, top_p=0.9):
    prompt = f"{emotion.lower()}: {start_text.lower()}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    while True:
        emotion = input("\n Duygu girin (örn: fear, happy, sad) (çıkmak için 'exit' yaz): ").strip()
        if emotion.lower() == "exit":
            print("Çıkış yapılıyor. Görüşmek üzere!")
            break
        start_tr = input(" Türkçe başlangıç cümlesi girin: ").strip()
        if start_tr.lower() == "exit":
            print("Çıkış yapılıyor. Görüşmek üzere!")
            break

        # Türkçe - İngilizce çeviri
        start_en = translate(start_tr, src_lang="tr", dest_lang="en")
        print("\n İngilizce Çeviri (başlangıç):", start_en)

        # İngilizce hikaye üretimi
        story_en = generate_story(emotion, start_en)
        print("\n Üretilen İngilizce Hikaye:\n", story_en)

        # İngilizce - Türkçe çeviri
        story_tr = translate(story_en, src_lang="en", dest_lang="tr")
        print("\n🇹🇷 Türkçe Çeviri:\n", story_tr)

        next_action = input("\n Başka bir hikaye üretmek ister misin? (evet/hayır): ").strip().lower()
        if next_action != "evet":
            print(" Çıkış yapılıyor. Görüşmek üzere!")
            break

if __name__ == "__main__":
    main()
