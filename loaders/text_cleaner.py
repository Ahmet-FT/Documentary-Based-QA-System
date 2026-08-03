import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Ham metni temizlemek için uygulanan tam pipeline.
    Aşağıdaki adımlar sırasıyla uygulanır:
      1. Unicode normalizasyonu (NFC)
      2. Tire ile bölünmüş kelimeleri yeniden birleştirme
      3. Kötü PDF satır sonlarını giderme (soft newlines)
      4. Birden fazla boşluk → tek boşluk
      5. Birden fazla boş satır → tek boş satır
      6. Baş/son boşlukları kırpma
    """
    if not text or not text.strip():
        return ""

    # 1. Unicode NFC normalizasyonu
    text = unicodedata.normalize("NFC", text)

    # 2. Tire ile bölünmüş kelimeleri birleştir (PDF'lerde sık rastlanan sorun)
    #    Örnek: "informa-\ntion" → "information"
    text = re.sub(r"-\n(\w)", lambda m: m.group(1), text)

    # 3. Soft newline'ları temizle:
    #    Paragraf sonu olmayan satır sonlarını (tek \n) boşlukla değiştir.
    #    Gerçek paragraf sonları (çift \n veya \n\n+) korunur.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 4. Birden fazla ardışık boşluk → tek boşluk
    text = re.sub(r"[ \t]+", " ", text)

    # 5. 3'ten fazla ardışık boş satır → 2 boş satır (paragraf ayrımını koru)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Baş/son boşlukları kırp
    text = text.strip()

    return text


def clean_text_lines(text: str) -> str:
    """
    Her satırı ayrı ayrı temizler (boş/anlamsız satırları atar).
    Satır başı/sonu boşlukları kırpar, tamamen boş satırları filtreler.
    Dipnot numaraları gibi gürültüyü kaldırır.
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Tamamen sayıdan oluşan kısa satırları atla (sayfa/dipnot numaraları)
        if re.fullmatch(r"\d{1,4}", line):
            continue

        # Sadece özel karakter içeren satırları atla
        if re.fullmatch(r"[\-_=\.\*\#~\s]+", line):
            continue

        cleaned_lines.append(line)

    # Birden fazla arka arkaya boş satırı teke indir
    result_lines = []
    prev_empty = False
    for line in cleaned_lines:
        if line == "":
            if not prev_empty:
                result_lines.append(line)
            prev_empty = True
        else:
            result_lines.append(line)
            prev_empty = False

    return "\n".join(result_lines).strip()


def full_clean(text: str) -> str:
    """
    clean_text + clean_text_lines'ı birleştiren tam temizlik fonksiyonu.
    Tüm loader'larda kullanılması önerilen ana fonksiyon.
    """
    text = clean_text(text)
    text = clean_text_lines(text)
    return text
