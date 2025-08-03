## Set up to work with format of The Motley Fool earnings transcirpts ##

import requests
from bs4 import BeautifulSoup


def Earnings_Call_Scraper(url, company, quarter, year):
    url = url
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/90.0.4430.93 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Get all paragraph elements
    paragraphs = soup.find_all("p")

    # Find where the actual transcript starts
    start_idx = None
    for i, p in enumerate(paragraphs):
        text = p.get_text(strip=True)
        if text.startswith("Operator") or "Prepared Remarks" in text:
            start_idx = i
            break

    # Extract from start index until "Duration:" is hit
    transcript_lines = []
    if start_idx is not None:
        for p in paragraphs[start_idx:]:
            text = p.get_text(strip=True)
            if text.startswith("Duration:"):
                break
            transcript_lines.append(text)

        transcript_text = "\n\n".join(transcript_lines)

        with open(f"{company}_{quarter}_{year}_transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript_text)

        print("✅ Transcript saved successfully and ended at 'Duration'.")
    else:
        print("❌ Could not find the start of the transcript.")

    return


# Call
Earnings_Call_Scraper( "https://www.fool.com/earnings/call-transcripts/2024/07/23/alphabet-googl-q2-2024-earnings-call-transcript/", "google", "Q2","2024")
