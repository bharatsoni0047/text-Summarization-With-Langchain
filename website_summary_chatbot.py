# IMPORTS
# Includes:
# - Streamlit UI components + chat features
# - URL validation
# - Groq LLM
# - Prompt templates
# - Webpage text loader

import streamlit as st
import validators

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


# LOAD GROQ API
groq_api_key = st.secrets["GROQ_API_KEY"]
branding = st.secrets.get("BRANDING", "")


# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Website Analyzer -- By Bharat",
    page_icon="🧠",
    layout="centered")


# CUSTOM CSS - DARK NEON CYBERPUNK THEME
st.markdown("""
<style>
    /* Dark background with futuristic image */
    .stApp {
        background: linear-gradient(rgba(10, 10, 30, 0.85), rgba(20, 0, 40, 0.95)),
                    url('https://static.vecteezy.com/system/resources/previews/026/716/487/large_2x/abstract-tech-lines-background-futuristic-abstract-shapes-technology-application-cover-and-web-site-design-generative-ai-illustration-free-photo.jpg') no-repeat center center fixed;
        background-size: cover;
        color: #e0e0ff;
    }

    /* Headers glow */
    h1, h2, h3 {
        color: #8B5CF6 !important;
        text-shadow: 0 0 15px rgba(139, 92, 246, 0.6);
        font-weight: bold;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 10px;
    }

    /* Buttons neon */
    .stButton > button {
        background: linear-gradient(45deg, #8B5CF6, #6366F1);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: bold;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.4);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.7);
    }

    /* Chat messages with glow */
    .stChatMessage {
        background-color: rgba(30, 20, 60, 0.6);
        border-radius: 15px;
        padding: 15px;
        margin: 15px 0;
        border: 1px solid #8B5CF6;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.2);
    }

    /* Input field */
    .stTextInput > div > div > input {
        background-color: rgba(20, 20, 50, 0.7);
        color: #e0e0ff;
        border: 2px solid #8B5CF6;
        border-radius: 12px;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: rgba(20, 20, 50, 0.8) !important;
        color: #e0e0ff;
    }

    /* Hide junk */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# MAIN TITLE
st.markdown('<div class="main-title">AI Website Analyzer</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Load any website & chat with its content intelligently</p>", unsafe_allow_html=True)


# MODEL SELECTION (TOP BAR STYLE)
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
        # --- Wikipedia handling ---
        if "wikipedia.org" in url:
            from langchain_community.document_loaders import WikipediaLoader

            if "search=" in url:
                query = url.split("search=")[1].split("&")[0]
            else:
                query = url.split("/")[-1].replace("_", " ")

            docs = WikipediaLoader(query=query, load_max_docs=1).load()

        # --- Normal attempt (static sites) ---
        else:
            from langchain_community.document_loaders import UnstructuredURLLoader

            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=False,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            docs = loader.load()

            # --- Fallback for JS-heavy sites (docs), not for every sites ---
            if not docs or len(docs[0].page_content.strip()) < 200:
                from langchain_community.document_loaders import PlaywrightURLLoader

                loader = PlaywrightURLLoader(
                    urls=[url],
                    remove_selectors=["nav", "footer", "header"]
                )
                docs = loader.load()

        if not docs:
            return None

        return " ".join(d.page_content for d in docs)

    except Exception as e:
        return None



# SESSION STATE SETUP
if "website_text" not in st.session_state:
    st.session_state.website_text = None
if "website_loaded" not in st.session_state:
    st.session_state.website_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "website_url" not in st.session_state:       
    st.session_state.website_url = ""


# URL INPUT FORM
st.markdown("---")
with st.form("url_form", clear_on_submit=True):
    url_input = st.text_input(
        "Enter Website URL to Analyze",
        placeholder="https://example.com"
    )
    load_button = st.form_submit_button("Load & Start Chatting 🧠")


# PROCESS URL WHEN LOADED
if load_button:
    if not url_input:
        st.error("Please enter a URL.")
    elif not validators.url(url_input):
        st.error("Invalid URL format.")
    else:
        with st.spinner("Analyzing website content..."):
            full_text = load_webpage_text(url_input)

        if not full_text:
            st.error("No readable content found on this website.")
        else:
            st.session_state.website_text = full_text
            st.session_state.website_url = url_input
            st.session_state.website_loaded = True
            st.session_state.chat_history = []
            st.success("Website loaded successfully! Ask anything below 👇")
            st.rerun()


# CHATBOT SETUP (only if website is loaded)
if st.session_state.website_loaded:

    st.markdown(
        f"""
        ### 🌐 **Current Website:**  
        **[{st.session_state.website_url}]({st.session_state.website_url})**
        """)
    st.markdown("---")

    # Chat history handler for LangChain
    history = StreamlitChatMessageHistory(key="chat_history")
    
    # LLM setup with selected model
    llm = ChatGroq(model=model_choice, groq_api_key=groq_api_key, temperature=0.4)

    # Main prompt template
    prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
You are a helpful assistant analyzing the following website content.
Answer questions accurately using only this content.
If asked for summary, give a clear overview.
If asked for details, provide long and thorough explanations.
Be friendly and natural.

IMPORTANT LANGUAGE RULE:
- Agar user pure English mein baat kar raha hai, toh full English mein reply kar.
- Agar user Hindi ya Hinglish mein baat kare (jaise "bhai", "yaar", gali, mazaak), tab hi casual Hindi-English mix mein reply kar.
- User ki language ke according hi switch kar – uska tone match kar.

Style when in Hinglish mode:
- Chill Indian bro vibe – short, fun, confident.
- Emoji situation ke hisaab se daal (zyada spam mat kar, 2-3 max per reply).
- Sarcasm samajh aur wapas de.
- Gali-mazaak pe friendly roast wapas kar, but over mat kar.
- Same answers repeat mat kar.

Examples (Hinglish mode):

1. User: "bhai ye design bilkul bakwas hai yaar"
   Reply: "Arre sorry bhai 😅 Socha pro banega lekin fail ho gaya. Ab bata kaisa chahiye exactly – dark, neon ya minimal? Jaldi fire version bana deta hu."

2. User: "waah bhai mast ban gaya!"
   Reply: "Thanks bro! Ab deploy kar ke dikha, sab jealous ho jayenge 😎 Aur kuch change chahiye toh bol dena."

3. User: "ye error aa raha hai bc fix kar na jaldi"
   Reply: "Haha calm down bc 😂 Yeh common error hai, 2 min mein fix – yeh line change kar aur push kar de. Ho gaya toh bol dena."

Example (Pure English mode):

User: "Can you summarize this website for me?"
Reply: "Sure! Here's a quick summary of the website: [clear English summary]. Let me know if you want more details on any part."

Bas itna daal de – ab bot smartly language switch karega, emoji bhi over nahi honge, aur vibe bilkul balanced rahegi 💪

Try kar ke bata "ab theek hai na bhai?" 😏
         
Website Content:
{st.session_state.website_text[:15000]}  # Limiting to avoid token overflow
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # Chain with history
    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Display chat history
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("human").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("ai").write(msg.content)

    # Chat input
    if user_input := st.chat_input("Ask anything about the website..."):
        st.chat_message("human").write(user_input)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = chain_with_history.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "abc123"}}
                )
            st.write(response.content)


# IF NO WEBSITE LOADED YET
else:
    st.info("👆 Enter a website URL above and click 'Load & Start Chatting' to begin analysis.")


# FOOTER
st.markdown("---")
st.caption(branding)

