# IMPORTS
# Includes:
# - Streamlit UI components + chat features
# - URL validation
# - Groq LLM
# - Prompt templates
# - Webpage text loader

import streamlit as st
import validators
from urllib.parse import unquote, parse_qs, urlparse

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader, WikipediaLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


# LOAD GROQ API
groq_api_key = st.secrets["GROQ_API_KEY"]
branding = st.secrets.get("BRANDING", "")


# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Website Analyzer -- By Bharat",
    page_icon="🧠",
    layout="centered"
)


# CUSTOM CSS - DARK + BLUE CYBERPUNK THEME
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, rgba(59,130,246,0.25), transparent 60%),
                linear-gradient(180deg, #050816, #0b1025);
    color: #e0e0ff;
}
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# MAIN TITLE
st.markdown('<h1 style="text-align:center;color:#8B5CF6;">AI Website Analyzer</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#aaa;'>Analyze any article or documentation page and chat with it</p>", unsafe_allow_html=True)


# MODEL SELECTION
col1, col2, col3 = st.columns([1,2,1])
with col2:
    model_choice = st.selectbox(
        "Select AI Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen/qwen3-32b"
        ],
        index=1
    )


# FUNCTION — Load webpage text
def load_webpage_text(url):
    """
    This function loads and extracts clean text content from a given webpage URL
    using UnstructuredURLLoader. It returns the combined text or None if no content.
    """
    try:
        url = url.split("#")[0]

        # --- Wikipedia handling ---
        if "wikipedia.org" in url:
            parsed_url = urlparse(url)

            if "search=" in parsed_url.query:
                query = parse_qs(parsed_url.query).get("search", [""])[0]
            elif "/wiki/" in parsed_url.path:
                query = unquote(parsed_url.path.split("/wiki/")[-1]).replace("_", " ")
            else:
                return None

            docs = WikipediaLoader(
                query=query,
                load_max_docs=1,
                doc_content_chars_max=12000
            ).load()

        # --- Normal static sites ---
        else:
            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=False,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            docs = loader.load()

        if not docs:
            return None

        return " ".join(d.page_content for d in docs)

    except Exception:
        return None


# SESSION STATE SETUP
if "website_text" not in st.session_state:
    st.session_state.website_text = ""
if "website_loaded" not in st.session_state:
    st.session_state.website_loaded = False
if "website_context_ready" not in st.session_state:
    st.session_state.website_context_ready = False
if "website_url" not in st.session_state:
    st.session_state.website_url = ""
if "no_content" not in st.session_state:
    st.session_state.no_content = False
if "content_valid" not in st.session_state:
    st.session_state.content_valid = False



# URL INPUT FORM
st.markdown("---")
with st.form("url_form", clear_on_submit=True):
    url_input = st.text_input("Enter Website URL", placeholder="https://example.com")
    load_button = st.form_submit_button("Load & Start Chatting 🧠")


# PROCESS URL
if load_button:
    if not url_input:
        st.error("Please enter a URL.")
    elif not validators.url(url_input):
        st.error("Invalid URL format.")
    else:
        with st.spinner("Analyzing website..."):
            full_text = load_webpage_text(url_input)

        st.session_state.website_text = full_text or ""
        st.session_state.website_url = url_input
        st.session_state.website_loaded = True
        st.session_state.website_context_ready = True

        if full_text and len(full_text.strip()) > 500:
            st.session_state.content_valid = True
            st.session_state.no_content = False
            st.success("Website loaded successfully! Ask anything below 👇")
        else:
            st.session_state.content_valid = False
            st.session_state.no_content = True
            st.info("Please enter this website into the chatbot.")


# CHATBOT
if st.session_state.website_loaded:

    st.markdown(f"""
    ### 🌐 Current Website  
    **[{st.session_state.website_url}]({st.session_state.website_url})**
    """)
    st.markdown("---")

    history = StreamlitChatMessageHistory(key="chat_history")

    if st.session_state.website_context_ready and not history.messages:
            if st.session_state.content_valid:
                history.add_ai_message("I’m ready 👍 Ask me anything about this website — what it is, what it does, or how it’s used.")
            else:
                history.add_ai_message(
                    "Please enter this website into the chatbot.")

    llm = ChatGroq(
        model=model_choice,
        groq_api_key=groq_api_key,
        temperature=0.4
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
If website content is empty AND no_content flag is true:
- Do NOT explain the website
- Do NOT guess
- Reply ONLY with:
  "Please enter this website into the chatbot."
     
You are an assistant chatting about a specific website.
The website URL is already known to you from context.
Always assume the conversation is about the current website unless the user clearly changes the topic.

WEBSITE CONTEXT RULES:
- You always know the website URL.
- If the user asks for the website name, infer it from the URL automatically.
- Never say that the user did not provide the website name.
- Never ask the user to share the website again.
- Never blame the user for missing information.

CONTENT RULES:
- If website content is available, answer strictly using that content.
- If website content is empty or incomplete, still explain what the website is using general public knowledge.
- Do NOT complain about missing content.
- Do NOT say things like “content not available” or “I only know general knowledge”.
- First explain what the website is, then add limitations only if truly needed.

LANGUAGE RULE (STRICT):
- If the user message is fully in English, reply ONLY in English.
- Do NOT use Hindi or Hinglish unless the user uses Hindi/Hinglish words first.
- Mixed language replies are allowed ONLY if the user mixes languages.
- Always match the user’s language exactly.

STYLE RULES:
- Be confident, calm, and helpful.
- Take initiative to explain instead of asking questions.
- Be concise but informative.
- Friendly tone, never defensive.
- Emojis only when appropriate (max 2–3).

Website Content:
{content}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")])

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda _: history,
        input_messages_key="input",
        history_messages_key="history"
    )

    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("human").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("ai").write(msg.content)

    if user_input := st.chat_input("Ask me about this website..."):
        st.chat_message("human").write(user_input)
        with st.chat_message("ai"):
            response = chain_with_history.invoke(
                {
                    "input": user_input,
                    "content": st.session_state.website_text[:15000] if st.session_state.content_valid else ""
                },
                config={"configurable": {"session_id": "website-chat"}}
            )
            st.write(response.content)

else:
    st.info("👆 Enter a website URL above to begin.")


# FOOTER
st.markdown("---")
st.caption(branding)