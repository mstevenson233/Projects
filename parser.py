import pandas as pd
import re

def parse_transcript_to_df(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    text = re.sub(r"\n+", "\n", raw_text)

    pattern = r"(?P<speaker>[A-Z][a-z]+(?: [A-Z][a-z]+)*|Operator)\s*--\s*.*?\n(?P<text>.*?)(?=(?:[A-Z][a-z]+(?: [A-Z][a-z]+)*|Operator)\s*--\s*|Duration:|$)"
    matches = re.finditer(pattern, text, flags=re.DOTALL)

    speaker_data = []
    for match in matches:
        speaker = match.group("speaker").strip()
        speech = match.group("text").strip()
        speaker_data.append({"speaker": speaker, "text": speech})

    return pd.DataFrame(speaker_data)