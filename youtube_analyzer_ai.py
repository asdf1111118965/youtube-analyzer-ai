import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import whisper
import re
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="🎥 YouTube Analysis AI Agent (No Login, No Payment)", page_icon="🤖", layout="centered")
st.title("🎥 YouTube Analysis AI Agent (No Login, No Payment)")
st.caption("💡 Works 100% online — Unlimited trials, no API key needed!")

# ---------- FUNCTIONS ----------

def get_video_id(url):
    """Extract YouTube video ID from URL."""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def fetch_transcript(video_id):
    """Fetch transcript text from YouTube with fallback."""
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join([t["text"] for t in transcript_data])
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None


@st.cache_resource
def load_whisper_model():
    """Load Whisper model (cached)."""
    return whisper.load_model("base")


from faster_whisper import WhisperModel

@st.cache_resource
def load_whisper_model():
    """Load lightweight Whisper model."""
    return WhisperModel("base", device="cpu")

def transcribe_with_whisper(url):
    """Transcribe YouTube audio when captions are unavailable."""
    st.info("🎧 Generating transcript using Faster Whisper — please wait...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '/tmp/audio.%(ext)s',
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".mp3")

    model = load_whisper_model()
    segments, info = model.transcribe(filename, beam_size=5)
    transcript = " ".join(segment.text for segment in segments)

    if os.path.exists(filename):
        os.remove(filename)

    return transcript


def summarize_text(transcript):
    """Summarize transcript using Sumy (lightweight)."""
    parser = PlaintextParser.from_string(transcript, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, 10)  # top 10 sentences

    summary = " ".join(str(sentence) for sentence in summary_sentences)

    formatted_summary = f"""
    **1. Heading:**  
    Overview of the video and its key discussion points.  

    **2. Topic and Sub-Topics:**  
    {summary.strip()}

    **3. Conclusion:**  
    The video provides insights and explanations relevant to the topic discussed.
    """
    return formatted_summary


def get_video_info(url):
    """Fetch video title and URL."""
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "Unknown Title")
            webpage_url = info.get("webpage_url", url)
            return title, webpage_url
    except Exception as e:
        st.warning(f"⚠️ Could not fetch video info: {e}")
        return "Unknown Title", url


def get_audio_download_url(url):
    """Generate a safe MP3 download link."""
    try:
        ydl_opts = {'format': 'bestaudio/best', 'quiet': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception:
        return None


# ---------- UI ----------
url = st.text_input("🔗 Enter YouTube Video URL:")
output_format = st.radio("🧾 Choose Output Format:", ["Summarise with text", "Summarise with audio"])

if st.button("🚀 Analyze Video"):
    if not url:
        st.warning("Please enter a YouTube link first.")
        st.stop()

    video_id = get_video_id(url)
    if not video_id:
        st.error("❌ Invalid YouTube URL.")
        st.stop()

    title, watch_url = get_video_info(url)
    st.subheader(f"🎬 {title}")

    # Step 1: Try transcript API
    st.write("⏳ Fetching transcript...")
    transcript = fetch_transcript(video_id)

    # Step 2: If not available, fallback to Whisper
    if not transcript:
        st.warning("⚠️ No captions found — switching to Whisper transcription.")
        try:
            transcript = transcribe_with_whisper(url)
        except Exception as e:
            st.error(f"❌ Whisper transcription failed: {e}")
            st.stop()

    # Step 3: Summarize
    if transcript:
        st.success("✅ Transcript ready! Generating summary...")
        summary = summarize_text(transcript)
        st.subheader("📜 Summary:")
        st.markdown(summary)
    else:
        st.error("❌ Could not generate transcript or summary for this video.")
        st.stop()

    # Step 4: Audio download
    st.markdown("---")
    st.subheader("🎵 Download Audio (MP3)")
    audio_url = get_audio_download_url(url)
    if audio_url:
        st.markdown(f"[⬇️ Click here to download MP3]({audio_url})")
    else:
        st.warning("⚠️ Unable to fetch a direct MP3 link.")
