# Storyteller AI

Duygu ve başlangıç cümlesinden hikâye üretmeye yönelik bir derin öğrenme projesi. Türkçe giriş İngilizceye çevrilir, GPT-2 Medium ile metin üretilir ve sonuç Türkçeye çevrilerek sunulur.

**Bu repo iki ayrı model bileşeni içerir:**

- **Aktif uygulama: hazır GPT-2 Medium.** Web arayüzü, Flask API ve komut satırı uygulaması `gpt2-medium` kullanır. Bu model proje kapsamında eğitilmemiş veya fine-tune edilmemiştir.
- **Proje kapsamında eğitilen özel Transformer.** PyTorch ile tanımlanan `StoryTransformer` mimarisi, eğitilmiş checkpoint, tokenizer, eğitim kodları ve kayıtlı sonuçlar `custom_transformer/` klasöründe bulunur. Bu model varsayılan uygulamaya bağlı değildir.

> Uygulamayı başlatmak özel Transformer'ı çalıştırmaz. Varsayılan hikâye üretimi GPT-2 Medium ile devam eder. GPT-2 ağırlıkları ilk kullanımda Hugging Face üzerinden indirilir; özel Transformer ağırlıkları ise repoya dahildir.

## Özellikler

- Duygu ve başlangıç metniyle yönlendirilen hikâye üretimi.
- Türkçe → İngilizce → Türkçe çeviri akışı.
- Flask web arayüzü ve JSON API.
- Etkileşimli komut satırı kullanımı.
- CUDA destekli GPU veya CPU üzerinde çalışma.
- Özel Transformer için eğitim kodları, kaydedilmiş ağırlıklar ve yükleme kontrolü.

## Kurulum

Python 3.12 ile kontrol edilmiştir. Projeyi indirip bir sanal ortam oluşturun:

```bash
git clone https://github.com/emrekilincc/storyteller-AI.git
cd storyteller-AI
python -m venv .venv
```

Sanal ortamı Windows PowerShell'de `.venv\Scripts\Activate.ps1`, macOS/Linux'ta `source .venv/bin/activate` komutuyla etkinleştirin.

```bash
python -m pip install -r requirements.txt
```

İlk model indirmesi için internet bağlantısı ve model dosyaları için yeterli disk alanı gerekir. Çeviri işlemleri `deep-translator` üzerinden Google çeviri servisine bağlanır; çevrimdışı çalışmaz ve servisin kullanım sınırlarına tabidir. Metin üretimi yerel bilgisayarda yapılır.

## Web arayüzü ve komut satırı

Web uygulamasını başlatın:

```bash
python arayuz.py
```

Tarayıcıda `http://127.0.0.1:5000` adresini açın. Duygu seçin, Türkçe başlangıç cümlesi girin ve **Hikaye Oluştur** düğmesine basın. Sunucu yerel geliştirme modunda çalışır; üretim ortamına dağıtım yapılandırması içermez.

Komut satırını tercih ederseniz:

```bash
python transformer_hikaye_uretim.py
```

Dosya adında “transformer” geçmesine rağmen bu uygulama **GPT-2 Medium** kullanır. Örnek duygu girdileri: `happy`, `sad`, `fear`. Duygu veya başlangıç alanına `exit` yazarak çıkabilirsiniz.

## API

`POST /api/generate` — `Content-Type: application/json`

```json
{
  "emotion": "happy",
  "start": "Bir sabah ormanda küçük bir kuş gördüm."
}
```

Yanıt, üretilen metni `story` alanında döndürür.

## Eğitilmiş özel Transformer

`custom_transformer/best_transformer_checkpoint.pth`, proje arşivinden alınmış yaklaşık **98,3 MiB** büyüklüğünde bir PyTorch ağırlık dosyasıdır. İlgili model tanımı ve tokenizer aynı arşivden alınmıştır.

| Parametre | Değer |
| --- | --- |
| Token sayısı, padding dahil | 39.880 |
| Embedding boyutu | 256 |
| Feed-forward boyutu | 512 |
| Attention head sayısı | 8 |
| Encoder / decoder katmanı | 1 / 6 |
| Maksimum giriş uzunluğu | 60 token |
| Duygu embedding boyutu | 1 kategori |

Eğitim kodu duyguyu metne önek olarak ekler; ayrı duygu embedding girdisine sabit `0` gönderir. `vocab.json` yardımcı sözlüğü tokenizer ile ayrı oluşturulmuştur; bu iki indeksleme doğrudan birbirinin yerine kullanılmamalıdır.

Checkpoint'i doğrulamak için:

```bash
python -m pip install -r requirements-transformer.txt
python custom_transformer/verify_checkpoint.py
```

Bu kontrol, ağırlıkları model tanımına eksiksiz yükler, tokenizer boyutunu doğrular ve kısa bir ileri hesaplama yapar. Eğitim başlatmaz ve GPT-2'yi indirmez. Hikâye kalitesini ölçmez.

### Eğitim dosyaları

Eğitim betiği ve hazırlanan veri dosyası aynı klasörde bulunur. Yeniden eğitim için:

```bash
cd custom_transformer
python train_transformer.py
```

Yeni checkpoint ve metrikler `custom_transformer/runs/` altında oluşturulur; arşivden alınan model ve sonuçlar korunur. Eğitim süresi donanıma bağlıdır. Bu düzenleme sırasında eğitim yeniden çalıştırılmamıştır.

`data_prepare_transformer.py` GoEmotions ve DailyDialog verilerini birleştirmeye, `tokenizer.py` tokenizer ve sözlük oluşturmaya yönelik özgün yardımcı betiklerdir. Veri seti servislerinin güncel şemalarıyla yeniden indirme akışı bu düzenlemede test edilmemiştir. Mevcut checkpoint'i kullanırken tokenizer'ı yeniden oluşturmayın.

`loss_log_transformer.csv` ve grafikler özgün arşivdeki eğitim kayıtlarıdır. Bunlar bağımsız bir hikâye kalitesi değerlendirmesi değildir. `provenance.json`, alınan dosyaların kaynak konumunu, boyutlarını ve SHA-256 özetlerini içerir.

## Proje yapısı

```text
storyteller-AI/
├── arayuz.py                         # Aktif GPT-2 web uygulaması ve API
├── transformer_hikaye_uretim.py       # Aktif GPT-2 komut satırı uygulaması
├── templates/index.html              # Web arayüzü
├── requirements.txt                  # GPT-2 uygulamasının bağımlılıkları
├── requirements-transformer.txt      # Özel Transformer için ek bağımlılıklar
└── custom_transformer/
    ├── model.py                      # StoryTransformer mimarisi
    ├── best_transformer_checkpoint.pth
    ├── tokenizer.pickle
    ├── vocab.json
    ├── verify_checkpoint.py
    ├── train_transformer.py
    ├── data_prepare_transformer.py
    ├── tokenizer.py
    ├── datasets/merged_transformer_dataset.csv
    ├── loss_log_transformer.csv
    ├── *_plot_transformer.png
    └── provenance.json
```

## Bilinen sınırlamalar

- GPT-2'ye duygu bilgisi bir metin öneki olarak verilir; duyguya uyum, metin tutarlılığı veya içerik uygunluğu garanti edilmez.
- Çeviri servisi hata verdiğinde mevcut kod özgün metinle devam eder; sonuç Türkçe olmayabilir.
- Son kontrolde Google çeviri servisi hız sınırı hatası verdi. Türkçe girişten Türkçe çıktıya kadar tüm canlı çeviri akışı doğrulanamadı.
- Özel Transformer deneysel bir eğitim çalışmasıdır. Ağırlıkların yüklenmesi ve sayısal çıktı üretmesi, genel hikâye kalitesinin doğrulandığı anlamına gelmez.
