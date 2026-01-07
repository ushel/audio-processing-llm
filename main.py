import google.genai as genai
import json
import os
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import random

# ================= ENV =================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# ================= HELPERS =================

def wait_for_file_active(client, file_name, timeout=180):
    print("Waiting for file processing...")
    start = time.perf_counter()

    while time.perf_counter() - start < timeout:
        file_info = client.files.get(name=file_name)
        if file_info.state.name == "ACTIVE":
            elapsed = time.perf_counter() - start
            print(f"File ready in {elapsed:.2f}s")
            return file_info, elapsed

        print(f"File state: {file_info.state.name}...")
        time.sleep(2)

    raise TimeoutError("File processing timed out")


def generate_with_retry(client, model, contents, max_retries=5):
    start = time.perf_counter()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents
            )
            elapsed = time.perf_counter() - start
            return response, elapsed

        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Gemini overloaded ({attempt+1}/{max_retries}), retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Max retries exceeded")


def extract_text_from_response(response):
    start = time.perf_counter()

    text = ""
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "text"):
                text += part.text

    elapsed = time.perf_counter() - start
    text = text.strip()

    print("Response preview:", text[:200] + ("..." if len(text) > 200 else ""))
    print(f"Text extraction: {elapsed:.2f}s")

    return text, elapsed


def extract_json_from_text(text):
    start = time.perf_counter()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]

    text = text.replace("```", "").strip()
    parsed = json.loads(text)

    elapsed = time.perf_counter() - start
    print(f"JSON parsing: {elapsed:.2f}s")

    return parsed, elapsed

# ================= CORE =================

def extract_schema_from_audio(audio_path: str):
    total_start = time.perf_counter()

    client = genai.Client(api_key=API_KEY)

    # -------- Upload audio --------
    print("Uploading audio...")
    upload_start = time.perf_counter()
    uploaded_file = client.files.upload(file=audio_path)
    upload_time = time.perf_counter() - upload_start
    print(f"Upload completed in {upload_time:.2f}s")

    # -------- File activation --------
    active_file, processing_time = wait_for_file_active(client, uploaded_file.name)

    # -------- Prompt --------
    prompt = """
Return ONLY valid JSON-LD (no explanation).

{
  "@context": "https://schema.org",
  "@type": "AudioObject"
}

Extract:
- Person (speakers)
- Product (mentioned brands/tools)
- Event (talks, meetings, announcements)
"""

    # -------- Gemini inference --------
    print("Calling Gemini with audio...")
    response, inference_time = generate_with_retry(
        client,
        model="gemini-2.5-flash",
        contents=[prompt, active_file]
    )
    print(f"Gemini inference: {inference_time:.2f}s")

    # -------- Response parsing --------
    response_text, text_time = extract_text_from_response(response)
    schema, json_time = extract_json_from_text(response_text)

    # -------- Metrics --------
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0

    total_cost = (input_tokens * 0.30 + output_tokens * 2.50) / 1_000_000
    total_time = time.perf_counter() - total_start

    schema["metrics"] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(total_cost, 6),
        "timings_seconds": {
            "upload": round(upload_time, 2),
            "file_processing": round(processing_time, 2),
            "inference": round(inference_time, 2),
            "text_extraction": round(text_time, 2),
            "json_parsing": round(json_time, 2),
            "total_execution": round(total_time, 2)
        }
    }

    print(f"\nTOTAL EXECUTION TIME: {total_time:.2f}s")
    return schema


# ================= MAIN =================

if __name__ == "__main__":
    audio_path = input("Audio path: ").strip() or "audio.wav"

    if not Path(audio_path).exists():
        print("Audio file not found")
        exit(1)

    if not API_KEY:
        print("GEMINI_API_KEY missing")
        exit(1)

    try:
        schema = extract_schema_from_audio(audio_path)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"schema_audio_{Path(audio_path).stem}_{ts}.jsonld"

        with open(output_file, "w") as f:
            json.dump(schema, f, indent=2)

        print(f"\nSaved → {output_file}")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Try shorter audio (≤60s) or retry after a few minutes")
