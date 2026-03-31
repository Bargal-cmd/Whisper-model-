from fastapi import FastAPI, UploadFile, File
import whisper
import os
import uuid
import traceback
import subprocess
import re
import torch
import torchaudio

from silero_vad import load_silero_vad, get_speech_timestamps

app = FastAPI()

# -----------------------------
# LOAD MODELS
# -----------------------------
print("🚀 Loading Whisper model...")
model = whisper.load_model("medium")   # better than base
print("✅ Whisper model loaded")

print("🚀 Loading Silero VAD...")
vad_model = load_silero_vad()
print("✅ Silero VAD loaded")

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu"
}

# -----------------------------
# AUDIO CONVERSION
# -----------------------------
def convert_to_wav(input_path, output_path):
    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# -----------------------------
# VAD: KEEP ONLY SPEECH
# -----------------------------
def apply_vad(input_wav, output_wav):
    """
    Simple FFmpeg-based trimming (instead of torchaudio VAD)
    This avoids TorchCodec issues.
    """

    command = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-af", "silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB",
        output_wav
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_wav)
    except:
        return False
    waveform, sample_rate = torchaudio.load(input_wav)

    # convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    speech_timestamps = get_speech_timestamps(
        waveform.squeeze(0),
        vad_model,
        sampling_rate=sample_rate
    )

    if not speech_timestamps:
        return False

    speech_chunks = []
    for ts in speech_timestamps:
        start = ts["start"]
        end = ts["end"]
        speech_chunks.append(waveform[:, start:end])

    if not speech_chunks:
        return False

    merged = torch.cat(speech_chunks, dim=1)
    torchaudio.save(output_wav, merged, sample_rate)
    return True

# -----------------------------
# REMOVE REPETITION
# -----------------------------
def remove_repeated_segments(segments):
    cleaned = []
    prev_text = ""

    for seg in segments:
        text = seg["text"].strip().lower()

        if not text:
            continue

        if text == prev_text:
            continue

        if len(text) < 2:
            continue

        cleaned.append(seg)
        prev_text = text

    return cleaned

# -----------------------------
# JOIN SEGMENTS
# -----------------------------
def join_segments(segments):
    return " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())

# -----------------------------
# CORRECTION LAYER
# -----------------------------
CORRECTIONS = {
    "mouchipu": "mujhe",
    "kaha hai": "bukhar hai",
    "pen": "pain",
    "witness": "weakness",
    "bhukarhe": "bukhar hai",
    "bukharhe": "bukhar hai",
    "body mein pen": "body mein pain",
    "weakness ho raha hai": "weakness ho rahi hai",
    "paira": "pairon",
    "sujan": "swelling"
}

def clean_text(text):
    text = text.lower()
    for wrong, correct in CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return text.strip()

# -----------------------------
# REMOVE GARBAGE / HALLUCINATION
# -----------------------------
def remove_garbage_tail(text):
    # repeated same token 4+ times
    match = re.search(r'(\b\w+\b)(?:\s+\1){3,}', text, flags=re.IGNORECASE)
    if match:
        text = text[:match.start()].strip()

    nonsense_patterns = [
        r'तो पेदन.*',
        r'पेदन.*',
        r'bhukarhe.*',
        r'haa haa haa.*'
    ]

    for pattern in nonsense_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text.strip()

# -----------------------------
# WHISPER SAFE TRANSCRIBE
# -----------------------------
def run_transcription(audio_path, task="transcribe", language=None):
    result = model.transcribe(
        audio_path,
        task=task,
        language=language,
        fp16=False,
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.8,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.0
    )
    return result

# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "Upgraded Whisper + Silero VAD API Running"}

# -----------------------------
# MAIN API
# -----------------------------
@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())

        raw_path = os.path.join(TEMP_DIR, f"{file_id}_{file.filename}")
        wav_path = os.path.join(TEMP_DIR, f"{file_id}_clean.wav")
        speech_only_path = os.path.join(TEMP_DIR, f"{file_id}_speech.wav")

        # Save uploaded file
        with open(raw_path, "wb") as f:
            f.write(await file.read())

        print(f"📁 Uploaded: {raw_path}")

        # Convert to standard WAV
        convert_to_wav(raw_path, wav_path)
        print(f"✅ Converted to WAV: {wav_path}")

        # Apply VAD
        has_speech = apply_vad(wav_path, speech_only_path)

        if not has_speech:
            os.remove(raw_path)
            os.remove(wav_path)
            return {
                "success": False,
                "error": "No clear speech detected in the audio."
            }

        print(f"✅ Speech-only file created: {speech_only_path}")

        # -----------------------------
        # ORIGINAL TRANSCRIPTION
        # -----------------------------
        transcribe = run_transcription(
            speech_only_path,
            task="transcribe",
            language="hi"   # force Hindi for better testing
        )

        detected_code = transcribe.get("language", "unknown")
        detected_name = LANGUAGE_MAP.get(detected_code, "Unknown")

        segments = [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            }
            for seg in transcribe.get("segments", [])
        ]

        segments = remove_repeated_segments(segments)

        original_text = join_segments(segments)
        original_text = remove_garbage_tail(original_text)

        cleaned_text = clean_text(original_text)
        cleaned_text = remove_garbage_tail(cleaned_text)

        # -----------------------------
        # ENGLISH TRANSLATION
        # -----------------------------
        translate = run_transcription(
            speech_only_path,
            task="translate",
            language=None
        )

        translated_segments = [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            }
            for seg in translate.get("segments", [])
        ]

        translated_segments = remove_repeated_segments(translated_segments)
        english_text = join_segments(translated_segments)
        english_text = remove_garbage_tail(english_text)

        # -----------------------------
        # CLEANUP TEMP FILES
        # -----------------------------
        for path in [raw_path, wav_path, speech_only_path]:
            if os.path.exists(path):
                os.remove(path)

        return {
            "success": True,
            "detected_language_code": detected_code,
            "detected_language_name": detected_name,
            "original_transcript": original_text,
            "cleaned_transcript": cleaned_text,
            "english_translation": english_text,
            "segments": segments,
            "translated_segments": translated_segments
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }