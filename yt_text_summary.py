# IMPORTS
# Includes:
# - Streamlit UI components
# - URL validation
# - Groq LLM + Whisper
# - Prompt templates
# - Webpage text loader
# - YouTube audio downloader

import streamlit as st
import validators
import time
import os
import subprocess

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredURLLoader

from groq import Groq


# LOAD GROQ API
groq_api_key = st.secrets["GROQ_API_KEY"]
branding = st.secrets.get("BRANDING", "")


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Smart URL Summarizer--By Bharat",
    page_icon="🚀",
    layout="centered"
)


# PAGE TITLE
st.markdown("""
# 🚀 Rocket Smart URL Summarizer
Summarize any YouTube video or Website into **clean, simple words**.

- Uses Groq + Llama models  
- Works in English + Hindi  
- Fast, clean and accurate  
""")


# SIDEBAR SETTINGS
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Choose Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen/qwen3-32b"
        ],
        index=1
    )

    style = st.radio("Summary Type", ["Normal", "Bullet Points", "Hindi Mein"])

    # ---------------- SUMMARY LENGTH MOVED TO SIDEBAR ---------------- #
    st.subheader("Summary Length")

    length_choice = st.radio(
        "Choose Length",
        [
            "Short (120-150 words)",
            "Medium (250-300 words)",
            "Long (400-500 words)"
        ],
        index=1
    )


# Summary length mapping
length_map = {
    "Short (120-150 words)": "120-150",
    "Medium (250-300 words)": "250-300",
    "Long (400-500 words)": "400-500",
}

selected_range = length_map.get(length_choice, "250-300")


# PROMPTS (Updated)
prompts = {

"Normal": f"""
You are an expert summarizer.

Write a **clear, human-like summary** in **{selected_range} words**.
Make it highly readable, well-structured, and accurate.

Follow these rules:
- Start with a short 1–2 sentence overview.
- Cover only the **main ideas**, no filler.
- Write in smooth natural English.
- Do NOT repeat points.
- Do NOT add extra opinions.

Content to summarize:
{{text}}
""",

"Bullet Points": f"""
Convert the content into **10–15 high-quality bullet points**.
The tone must be clear, compact, and meaningful.

Rules:
- Each bullet must express **one strong idea**.
- Bullets should not repeat each other.
- Keep output near **{selected_range} words**.
- Use simple English anyone can understand.

Content:
{{text}}
""",

"Hindi Mein": f"""
❗IMPORTANT: Write ONLY in Hindi (Devanagari script).  
Punjabi, Hinglish, or any other script is NOT allowed.

नीचे दिए गए कंटेंट को **{selected_range} शब्दों** में  
बहुत ही आसान, साफ़ और पूरी तरह हिंदी (देवनागरी) में लिखो।

Rules:
- शुरुआत एक छोटे ओवरव्यू से करो।
- केवल मुख्य बिंदु लिखो।
- भाषा 8th–10th class की आसान हिंदी हो।
- किसी भी तरह का पंजाबी शब्द नहीं होना चाहिए।
- Hinglish नहीं लिखना (जैसे: “yaad”, “pyaar”, “raat” नहीं — “याद”, “प्यार”, “रात” लिखो)।
- Repetition नहीं करना।

Content:
{{text}}
"""

}

prompt_template = PromptTemplate(
    template=prompts[style],
    input_variables=["text"]
)



# FUNCTION 1 — YouTube → Whisper Transcript
def transcribe_with_whisper(url, api_key):
    subprocess.run(
        ["yt-dlp", "-f", "bestaudio", "-o", "audio.%(ext)s", url],
        check=True
    )

    audio_file = next((f for f in os.listdir() if f.startswith("audio.")), None)
    if not audio_file:
        raise FileNotFoundError("Audio file not downloaded")

    client = Groq(api_key=api_key)

    with open(audio_file, "rb") as f:
        transcript_text = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3"
        ).text

    os.remove(audio_file)
    return transcript_text



# FUNCTION 2 — Load webpage text
def load_webpage_text(url):
    loader = UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    docs = loader.load()

    if not docs:
        return None

    return " ".join(d.page_content for d in docs)



# MAIN SUMMARIZER ENGINE
def summarize_with_retry(url, api_key, model_name, max_retries=3):

    llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.4)

    for attempt in range(max_retries):
        try:
            with st.spinner(f"Processing... (Attempt {attempt+1})"):

                if "youtube.com" in url:
                    full_text = transcribe_with_whisper(url, api_key)
                else:
                    full_text = load_webpage_text(url)
                    if not full_text:
                        st.error("No readable content found.")
                        return None

                parser = StrOutputParser()
                chain = prompt_template | llm | parser

                return chain.invoke({"text": full_text})

        except Exception as e:
            error = str(e).lower()

            if "rate limit" in error:
                st.warning("Rate limit reached. Retrying...")
                time.sleep(8)

            elif "invalid" in error:
                st.error("Your Groq API key is invalid.")
                return None

            elif attempt == max_retries - 1:
                st.error(f"Failed after all retries. Error: {e}")

            else:
                st.warning("Retrying...")
                time.sleep(3)

    return None



# INPUT FORM
with st.form("summary_form"):
    url_input = st.text_input(
        "Enter YouTube or Website URL",
        placeholder="https://youtube.com/watch?v=xxxx"
    )

    submit = st.form_submit_button("Generate Summary")



# SUMMARY OUTPUT
if submit:
    # Auto-remove playlist parameters from YouTube links
    if "youtube.com" in url_input and "&list=" in url_input:
         url_input = url_input.split("&list=")[0]

    elif not url_input:
        st.error("Please enter a URL.")

    elif not validators.url(url_input):
        st.error("Invalid URL format.")

    else:
        summary = summarize_with_retry(url_input, groq_api_key, model_choice)

        if summary:
            st.success("Summary created successfully!")
            st.markdown("### 📄 Summary Output")
            st.write(summary)

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )

            with col2:
                st.code(summary)

            st.balloons()



# FOOTER
st.markdown("---")
st.caption(branding)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
