###############################
# IMPORTS
###############################

import streamlit as st
import validators
import time
import os
import subprocess

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredURLLoader

from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq


###############################
# LOAD GROQ API (hidden, secure)
###############################

groq_api_key = st.secrets["GROQ_API_KEY"]



###############################
# PAGE UI SETTINGS
###############################

st.set_page_config(
    page_title="Smart URL Summarizer",
    page_icon="rocket",
    layout="centered"
)


###############################
# PAGE TITLE
###############################

st.markdown("""
# Rocket Smart URL Summarizer
Summarize YouTube videos or Websites in **simple 300 words**.

- Very fast (Groq + Llama-3)
- Works in Hindi and English
""")


###############################
# SIDEBAR SETTINGS
###############################

with st.sidebar:
    st.header("Settings")

    # NO API INPUT NOW (REMOVED)

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



###############################
# PROMPT TEMPLATES
###############################

branding = st.secrets.get("BRANDING", "")

prompts = {
    "Normal": f"""
Write a simple 250-300 word summary in English.
Avoid repetition and cover only the key points.

At the end of the summary, add this line exactly:
{branding}

Content:
{{text}}
""",

    "Bullet Points": f"""
Summarize the content in 10-15 short and unique bullet points.
Focus only on important ideas.

At the end of the summary, add this line exactly:
{branding}

Content:
{{text}}
""",

    "Hindi Mein": f"""
Neeche diya content 250-300 shabdon mein simple Hindi mein summarize karo.

Ant mein ye line exactly add karna:
{branding}

Content:
{{text}}
"""
}


prompt_template = prompts[style]
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])



###############################
# YOUTUBE → WHISPER TRANSCRIBE
###############################

def transcribe_with_whisper(url, api_key):
    """
    Downloads YouTube audio → sends to Groq Whisper → returns transcript text.
    """

    # Download audio
    subprocess.run(
        ["yt-dlp", "-f", "bestaudio", "-o", "audio.%(ext)s", url],
        check=True
    )

    # Find audio file
    audio_file = next((f for f in os.listdir() if f.startswith("audio.")), None)
    if not audio_file:
        raise FileNotFoundError("Audio file not downloaded")

    # Transcribe with Whisper
    client = Groq(api_key=api_key)
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3"
        ).text

    os.remove(audio_file)
    return transcript



###############################
# WEBPAGE TEXT LOADER
###############################

def load_webpage_text(url):
    """
    Loads text from a normal webpage and returns all readable text.
    """

    loader = UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    docs = loader.load()
    if not docs:
        return None

    return " ".join(d.page_content for d in docs)



###############################
# MAIN SUMMARIZER ENGINE
###############################

def summarize_with_retry(url, api_key, model_name, max_retries=3):
    """
    Handles YouTube + Website.
    Retries on temporary API or network failures.
    Returns final summary text.
    """

    llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.4)

    for attempt in range(max_retries):
        try:
            with st.spinner(f"Loading content... (Attempt {attempt+1})"):

                # If YouTube link → use Whisper
                if "youtube.com" in url:
                    full_text = transcribe_with_whisper(url, api_key)

                # Else normal webpage text
                else:
                    full_text = load_webpage_text(url)
                    if not full_text:
                        st.error("No content found on this webpage.")
                        return None

                parser = StrOutputParser()
                chain = prompt | llm | parser

                return chain.invoke({"text": full_text})

        except Exception as e:
            err = str(e).lower()

            if "rate limit" in err:
                st.warning("Rate limit hit, retrying...")
                time.sleep(8)

            elif "invalid" in err:
                st.error("Your Groq API key is invalid.")
                return None

            elif attempt == max_retries - 1:
                st.error(f"Failed after all retries. Error: {e}")

            else:
                st.warning("Retrying...")
                time.sleep(3)

    return None



###############################
# URL INPUT BOX — ENTER → SUBMIT
###############################

with st.form("summary_form"):
    generic_url = st.text_input(
        "Enter YouTube or Website URL",
        placeholder="https://youtube.com/watch?v=xxxx"
    )

    submit = st.form_submit_button("Generate Summary")



###############################
# SUMMARY GENERATION
###############################

if submit:

    if not generic_url:
        st.error("Please enter a URL.")

    elif not validators.url(generic_url):
        st.error("URL is not correct.")

    else:
        summary = summarize_with_retry(generic_url, groq_api_key, model_choice)

        if summary:
            st.success("Summary created successfully!")
            st.markdown("### Summary Output")
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



###############################
# FOOTER
###############################

st.markdown("---")
st.caption("Made with ❤️ | Powered by Groq + Llama Models")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)




