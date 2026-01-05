# ingest.py → PDF → chunks
import pdfplumber
import unicodedata

def fix_pdf_mojibake(text: str) -> str:
    # Unicode canonical normalization
    text = unicodedata.normalize("NFKC", text)

    # Explicit PDF mojibake fixes
    replacements = {
        "â€“": "–",
        "â€”": "—",
        "â€™": "’",
        "â€œ": "“",
        "â€�": "”",
        "â†’": "→",
        "â‰¥": "≥",
        "â‰¤": "≤",
        "Âµ": "µ",
        "Â°": "°",
        "Ã—": "×",
        "Ã·": "÷",
        "Â±": "±",
        "Â²": "²",
        "Â³": "³",
        "â€¢": "•",
        "âˆ‘": "∑",
        "âˆš": "√",
    }


    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text

# Function to load PDF and extract text
def load_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        previous_page_text = ""

        for page in pdf.pages:
            page_text = page.extract_text() or ""

            page_text = unicodedata.normalize("NFKC", page_text)
            page_text = fix_pdf_mojibake(page_text)

            normalized_text = " ".join(page_text.split())

            if len(normalized_text) < 50:
                continue

            if normalized_text == previous_page_text:
                continue

            text += normalized_text + "\n"
            previous_page_text = normalized_text

    return text


def chunk_texts(text, chunk_size=500, overlap=50, max_chunks=1000):
    chunks = {}
    start = 0
    text_length = len(text)
    chunk_id = 0

    while start < text_length and chunk_id < max_chunks:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks[chunk_id] = chunk
        chunk_id += 1

        if end == text_length:
            break

        start = end - overlap if overlap > 0 else end

    return chunks