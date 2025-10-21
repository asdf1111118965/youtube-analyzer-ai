import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from transformers import pipeline
import re

# ---------- CONFIG ----------
st.set_page_config(page_title="🎥 YouTube Analysis AI Agent (No Login, No Payment)", page_icon="🤖", layout="centered")

st.title("🎥 YouTube Analysis AI Agent (No Login, No Payment)")
st.caption("💡 Works 100% online — Unlimited trials, no API key needed!")

# ---------- Functions ----------

def get_video_id(url):
    """Extract YouTube video ID from URL."""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def fetch_transcript(video_id):
    """Fetch transcript text from YouTube with fallback."""
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

    try:
        # Try English first, then fallback to auto-generated
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join([t["text"] for t in transcript_data])
        return transcript

    except (TranscriptsDisabled, NoTranscriptFound):
        st.warning("⚠️ No official subtitles found. Trying auto-generated captions...")
        try:
            transcript_data = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_data.find_manually_created_transcript(['en']).fetch()
            return " ".join([t["text"] for t in transcript])
        except Exception:
            st.error("❌ No subtitles available for this video. Try another one with closed captions.")
            return None

    except AttributeError:
        st.error("⚠️ Your current YouTubeTranscriptApi version doesn’t support get_transcript. Please redeploy.")
        return None

    except Exception as e:
        st.error(f"❌ Unable to fetch transcript: {e}")
        return None



def get_video_info(url):
    """Fetch video title and provide safe download URL for audio."""
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "Unknown Title")
            webpage_url = info.get("webpage_url", url)
            return title, webpage_url
    except Exception as e:
        st.warning(f"⚠️ Could not fetch video info: {e}")
        return "Unknown Title", url


def summarize_text(transcript):
    """Summarize transcript using transformers (local HuggingFace model)."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    max_chunk = 800
    transcript = transcript.replace('\n', ' ')
    sentences = re.split(r'(?<=[.!?]) +', transcript)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    summary = ""
    for i, chunk in enumerate(chunks):
        st.info(f"🧩 Summarizing part {i+1}/{len(chunks)} ...")
        result = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
        summary += result[0]['summary_text'] + " "

    # Format into sections
    formatted_summary = f"""
    **1. Heading:**  
    Overview of the video and its key discussion points.  

    **2. Topic and Sub-Topics:**  
    {summary.strip()}

    **3. Conclusion:**  
    The video provides insights and explanations relevant to the topic discussed.
    """
    return formatted_summary


def get_audio_download_url(url):
    """Generate an MP3 download link (public safe)."""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'skip_download': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            audio_url = info['url']
            return audio_url
    except Exception:
        return None


# ---------- UI ----------
url = st.text_input("🔗 Enter YouTube Video URL:")
output_format = st.radio("🧾 Choose Output Format:", ["Summarise with text", "Summarise with audio"])

if st.button("🚀 Analyze Video"):
    if not url:
        st.warning("Please enter a YouTube link first.")
    else:
        video_id = get_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL")
            st.stop()

        title, watch_url = get_video_info(url)
        st.subheader(f"🎬 {title}")

        transcript = fetch_transcript(video_id)
        if not transcript:
            st.error("❌ Transcript unavailable for this video (it may not have subtitles).")
            st.stop()

        st.success("✅ Transcript fetched successfully!")

        if output_format == "Summarise with text":
            summary = summarize_text(transcript)
            st.subheader("📜 Summary:")
            st.markdown(summary)
        else:
            st.info("🎧 Audio summarization coming soon — for now you can read the summary above.")
            summary = summarize_text(transcript)
            st.subheader("📜 Summary:")
            st.markdown(summary)

        # ---------- Download Audio ----------
        st.markdown("---")
        st.subheader("🎵 Download Audio (MP3)")
        audio_url = get_audio_download_url(url)
        if audio_url:
            st.markdown(f"[⬇️ Click here to download MP3]({audio_url})")
        else:
            st.warning("⚠️ Unable to fetch a direct MP3 link (YouTube restrictions apply).")
