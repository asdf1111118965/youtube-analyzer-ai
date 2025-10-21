import streamlit as st
import yt_dlp
import librosa
import os
import re
import numpy as np
import subprocess
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

# ---------- CONFIG ----------
st.set_page_config(page_title="🎥 YouTube Analyzer AI (Free)", page_icon="🤖", layout="centered")

# ---------- Load Models ----------
@st.cache_resource
def load_whisper():
    model_name = "openai/whisper-tiny"  # Fast; use 'small' for better accuracy
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    return processor, model

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# ---------- Function: Download YouTube Audio ----------
def download_audio(link):
    st.info("🎬 Downloading audio from YouTube...")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "video_audio",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "postprocessor_args": [
            "-ar", "16000"
        ],
        "prefer_ffmpeg": True,
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

    expected_file = "video_audio.wav"
    if not os.path.exists(expected_file):
        for f in os.listdir():
            if f.startswith("video_audio") and f.endswith(".wav"):
                os.rename(f, expected_file)
                break

    if not os.path.exists(expected_file):
        raise FileNotFoundError("Audio file not created. Check ffmpeg or yt-dlp output.")

    return expected_file

# ---------- Function: Transcribe Audio (Chunked) ----------
def transcribe_audio(audio_file):
    st.info("🎧 Transcribing full audio locally (chunked Whisper)...")
    processor, model = load_whisper()
    audio, sr = librosa.load(audio_file, sr=16000)

    chunk_length = 30 * 16000  # 30-second chunks
    chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

    full_transcript = ""
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        st.write(f"🔊 Processing chunk {i+1}/{total_chunks}...")
        input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        full_transcript += " " + chunk_text

    return full_transcript.strip()

# ---------- Function: Summarize Long Text ----------
def summarize_long_text(text, max_chunk_len=3000):
    summarizer = load_summarizer()
    text = re.sub(r"\s+", " ", text).strip()
    chunks = [text[i:i + max_chunk_len] for i in range(0, len(text), max_chunk_len)]
    full_summary = ""

    for i, chunk in enumerate(chunks):
        st.write(f"🧩 Summarizing section {i+1}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=300, min_length=80, do_sample=False)[0]['summary_text']
        full_summary += " " + summary

    return full_summary.strip()

# ---------- Function: Analyze Transcript ----------
def analyze_content(transcript, style="Formal"):
    st.info("🧠 Analyzing content using free text summarizer...")

    text = re.sub(r"[^A-Za-z0-9.,?! ]+", " ", transcript)
    text = re.sub(r"\s+", " ", text).strip().capitalize()

    summary = summarize_long_text(text)
    summary = ". ".join([s.strip().capitalize() for s in summary.split('.') if s.strip()])

    structured_output = f"""
    **1️⃣ Title / Topic:**  
    {summary.split('.')[0].strip() if summary else 'N/A'}

    **2️⃣ Main Themes:**  
    - {summary.split('.')[1].strip() if len(summary.split('.')) > 1 else ''}
    - {summary.split('.')[2].strip() if len(summary.split('.')) > 2 else ''}

    **3️⃣ Key Facts:**  
    Derived from summarized context.

    **4️⃣ Notable Quotes:**  
    Not available (local Whisper output).

    **5️⃣ Summary ({style} style):**  
    {summary.strip()}
    """
    return structured_output

# ---------- Function: Create Podcast (macOS say + ffmpeg) ----------
import subprocess
import re

def create_podcast(text):
    st.info("🎙️ Generating podcast narration using macOS voice (say)...")

    # Convert structured text into a natural narration
    spoken_text = re.sub(r"\*\*.*?Title.*?:\*\*", "Let's begin with the main topic:", text)
    spoken_text = re.sub(r"\*\*.*?Main Themes.*?:\*\*", "The main themes discussed are:", spoken_text)
    spoken_text = re.sub(r"\*\*.*?Key Facts.*?:\*\*", "Here are some key facts:", spoken_text)
    spoken_text = re.sub(r"\*\*.*?Notable Quotes.*?:\*\*", "Some notable mentions:", spoken_text)
    spoken_text = re.sub(r"\*\*.*?Summary.*?\*\*", "In summary:", spoken_text)

    # Remove extra markdown characters
    spoken_text = re.sub(r"\*\*|[#`•\-:]", "", spoken_text)
    spoken_text = spoken_text.replace("1️⃣", "First,").replace("2️⃣", "Next,").replace("3️⃣", "Then,")

    mp3_file = "podcast_output.mp3"
    aiff_file = "podcast_temp.aiff"

    # Shorten overly long text to avoid truncation
    safe_text = spoken_text[:4000]

    # Generate AIFF using macOS 'say'
    subprocess.run(["say", "-o", aiff_file, safe_text])

    # Convert AIFF → MP3
    subprocess.run(["ffmpeg", "-y", "-i", aiff_file, mp3_file],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(aiff_file):
        os.remove(aiff_file)

    if os.path.exists(mp3_file) and os.path.getsize(mp3_file) > 1000:
        return mp3_file
    else:
        raise FileNotFoundError("⚠️ Podcast file creation failed.")

# ---------- STREAMLIT APP ----------
st.title("🎥 YouTube Analysis AI Agent (Free Version)")
st.markdown("💡 Works 100% offline — no OpenAI key, no payments required.")

link = st.text_input("🔗 Enter YouTube Video URL:")
output_format = st.radio("🧾 Choose Output Format:", ["Formal Text", "Podcast Style"])

if st.button("🚀 Analyze Video"):
    if not link:
        st.warning("Please enter a YouTube link first.")
    else:
        try:
            audio = download_audio(link)
            st.audio(audio)  # 🎧 play extracted YouTube audio

            transcript = transcribe_audio(audio)
            st.text_area("📝 Transcript Preview:", transcript[:1500] + "..." if len(transcript) > 1500 else transcript, height=200)

            style = "Formal" if output_format == "Formal Text" else "Conversational Podcast"
            result = analyze_content(transcript, style)

            if output_format == "Podcast Style":
                podcast_path = create_podcast(result)
                st.success("✅ Podcast created!")
                st.audio(podcast_path, format="audio/mp3")
            else:
                st.success("✅ Analysis complete!")
                st.subheader("📜 Structured Insights")
                st.markdown(result)

        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            try:
                if os.path.exists("video_audio.wav"):
                    os.remove("video_audio.wav")
                if os.path.exists("podcast_output.mp3"):
                    os.remove("podcast_output.mp3")
            except:
                pass
