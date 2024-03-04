import streamlit as st
import speech_recognition as sr
import google.generativeai as palm
import pyttsx3 
# initialisation 
engine = pyttsx3.init() 
palm.configure(api_key='AIzaSyCw9UHFLxolOl9fEBLnwFedqMBC6Sj8nPk')
# Use the palm.list_models function to find available models:
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name
def solution(name):
    prompt = name
    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0,
        # The maximum length of the response
        max_output_tokens=100,
    )
    result=completion.result 
    return result


def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        st.write("Recognizing...")
        user_input = recognizer.recognize_google(audio)
        return user_input
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand what you said.")
        return ""
    except sr.RequestError as e:
        st.write("Sorry, could not request results; {0}".format(e))
        return ""

def voice():
    st.title("Voice Assistant")
    user_input = None  # Initialize user_input variable outside the button blocks
    if st.button("speak"):
        user_input = listen()
        if user_input:  # Check if user_input is not None
            st.write("You said:", user_input)
            engine.say(solution(user_input))
            engine.runAndWait()
            st.write("Assistant:",solution(user_input)) 
        else:
            st.write("i can't help you")

if __name__ == "__main__":
    voice()
