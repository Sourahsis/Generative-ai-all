import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from googletrans import Translator
from langdetect import detect
from PIL import Image
import speech_recognition as sr
import google.generativeai as palm
translator = Translator()
load_dotenv()
apikey="AIzaSyCw9UHFLxolOl9fEBLnwFedqMBC6Sj8nPk"
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question,language):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    response=str(response["output_text"])
    response=translator.translate(response, dest=language).text
    print(response)
    st.write("Rahu: ", response)

def detect_language_code(text):
   language = detect(text)
   return language

def chat_with_pdf():
    st.header("Ask a Question from the PDF Files in eny language")
    user_question = st.text_input("Ask a Question")
    if(user_question):
        user_question = str(user_question) 
        language=detect_language_code(user_question)
        user_question=translator.translate(user_question, dest='en').text
    if user_question:
        user_input(user_question,language)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs=None
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
           if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
           else:
                st.write("please upload file")



def get_gemini_response(input,image,prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input,image[0],prompt])
    return response.text
    

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    try:
        if uploaded_file is not None:
            # Read the file into bytes
            bytes_data = uploaded_file.getvalue()

            image_parts = [
                {
                    "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                    "data": bytes_data
                }
            ]
            return image_parts
    except:
        st.write("please upload a file")


def chat_with_image():
    st.header("chat in any language , and the response will give in that language")
    input=st.text_input("Input Prompt: ",key="input")
    if(input):
        input = str(input) 
        language=detect_language_code(input)
        input=translator.translate(input, dest='en').text
    submit=st.button("submit")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)



    input_prompt = """
                You are an expert in understanding image text.
                You will receive input images , extract text &
                you will have to answer questions based on the input image text
                """

    ## If ask button is clicked
    if submit:
        if(input):
            image_data = input_image_setup(uploaded_file)
            response=get_gemini_response(input_prompt,image_data,input)
            response=translator.translate(response, dest=language).text
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("please ask something")

def chatbot():
    ## function to load Gemini Pro model and get repsonses
    model=genai.GenerativeModel("gemini-pro") 
    chat = model.start_chat(history=[])
    def get_gemini_response(question):
        response=chat.send_message(question,stream=True)
        return response

    ##initialize our streamlit app


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

def main():
    st.set_page_config("Chat PDF")
    page = st.sidebar.selectbox("Menu", ["Chat with pdf", "Chat with image","chatbot"])

    if page == "Chat with pdf":
        chat_with_pdf()
    elif page == "Chat with image":
        chat_with_image()
    elif page == "chatbot":
        chatbot()
if __name__ == "__main__":
    main()
