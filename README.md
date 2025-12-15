🚀 AI Website Analyzer — by Bharat

AI Website Analyzer is a smart, AI-powered Streamlit application that analyzes and summarizes websites, articles, and documentation pages into clear, structured, and easy-to-understand text.
The app is built to handle real-world websites with dynamic content, smart fallbacks, and language-aware responses.

It uses Groq Llama models for ultra-fast inference and LangChain for prompt orchestration, making it ideal for learning, research, and quick understanding of online content.


🔗 Live Demo
https://text-summarization-with-langchain-for-website-by-bharat.streamlit.app/


>>>>> Key Features

🌐 Website & Article Analysis
Extracts readable content from webpages using Unstructured loaders.

⚡ Ultra-Fast AI Responses
Powered by Groq Llama models for low-latency, high-quality answers.

🧠 Interactive Website Chatbot
Ask questions directly about the loaded website content.

🌍 Multilingual Support
Supports English and Hindi, automatically matching user language.

🧩 Smart Fallback Logic
If a website cannot be fully read, the chatbot still activates with clear guidance.

🎛️ Dynamic & User-Friendly UI
Clean Streamlit interface with custom styling and branding protection.

🛡️ Robust Error Handling
Gracefully handles dynamic, restricted, or partially readable websites.


>>>>> Tech Stack

Framework: Streamlit

LLMs: Groq (Llama models)

Prompt Orchestration: LangChain

Web Content Extraction: Unstructured Loader

Backend Language: Python

Deployment: Streamlit Cloud


 >>>>>What I Built

A dynamic website content extraction pipeline

Context-aware chatbot that remembers the current website

Fast and reliable AI responses using Groq LLMs

Language-aware reply system (English & Hindi)

Secure API handling via Streamlit secrets

Production-ready deployment on Streamlit Cloud


>>>>> Use Cases

Understanding long articles quickly

Exploring documentation pages

Research and content analysis

Learning from online resources efficiently


>>>>> Challenges I Faced & How I Solved Them:-

Some websites were not readable (dynamic / JS-heavy pages)
→ Used fallback logic and allowed the chatbot to activate even when content was missing.

Wikipedia links behaved differently (search, main page, article URLs)
→ Added custom Wikipedia URL parsing and used WikipediaLoader instead of normal loaders.

Chatbot said “URL not provided” even after entering a website
→ Passed the website URL explicitly into the system prompt instead of assuming context.

Bot gave random or generic answers when content was empty
→ Added a content validation flag and controlled fallback responses.

Chatbot activated but behaved inconsistently
→ Added an automatic first AI message to anchor the conversation context.

Language switching issues (English vs Hindi)
→ Enforced strict language rules based on user input language.

Streamlit app went to sleep on deployment
→ Understood cold-start behavior of Streamlit Cloud and accepted it as expected.

Users were confused when a website couldn’t be analyzed
→ Displayed a clear instruction message instead of error or guessing.

Prompt became too complex and confusing
→ Simplified and structured the prompt with clear rules and priorities.

Maintaining chat memory across messages
→ Used StreamlitChatMessageHistory with RunnableWithMessageHistory.
