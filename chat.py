from dotenv import load_dotenv
load_dotenv() ## loading all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from googletrans import Translator
from langdetect import detect
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
translator = Translator()

def detect_language_code(text):
   language = detect(text)
   return language

def chatbot():
    ## function to load Gemini Pro model and get repsonses
    model=genai.GenerativeModel("gemini-pro") 
    chat = model.start_chat(history=[])
    def get_gemini_response(question):
        response=chat.send_message(question,stream=True)
        return response

    ##initialize our streamlit app

    st.set_page_config(page_title="Q&A Demo")

    st.header("Chatbot Application")
    st.write("you can write in any language you want and the bot will response int those language")
    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    input1=st.text_input("Input: ",key="input")
    input=input1
    if(input):
        input = str(input) 
        language=detect_language_code(input)
        input=translator.translate(input, dest='en').text

    submit=st.button("Ask the question")
    if submit and input:
        response=get_gemini_response(input)
        # Add user query and response to session state chat history
        st.session_state['chat_history'].append(("You", input1))
        st.subheader("The Response is")
        for chunk in response:
            st.write(translator.translate(chunk.text, dest=language).text)
            st.session_state['chat_history'].append(("Bot", translator.translate(chunk.text, dest=language).text))
    st.subheader("The Chat History is")
        
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")
if __name__ == "__main__":
    chatbot()