
from tkinter import *
from tkinter import ttk,messagebox
import googletrans
import textblob
from googletrans import Translator

# Define a dictionary to map language names to language codes
languages = {
    "Spanish": 'es',
    "Chinese": 'zh-CN',
    "bengali": 'bn',
    "german": 'de',
    "hindi": 'hi',
    "Italian": 'it'
}
text="Anatomy is the identification and description of the structures of living things. It is a branch of biology and medicine. People who study anatomy study the body, how it is made up, and how it works. The study of anatomy dates back more than 2,000 yearsTrusted Source, to the Ancient Greeks. There are three broad areas: human anatomy animal anatomy — zootomy plant anatomy — phytotomy Human anatomy is the study of the structures of the human body. An understanding of anatomy is key to the practice of medicine and other areas of health. The word “anatomy” comes from the Greek words “ana,” meaning “up,” and “tome,” meaning “a cutting.” Traditionally, studies of anatomy have involved cutting up, or dissecting, organisms. Now, however, imaging technology can show us much about how the inside of a body works, reducing the need for dissection. Below, learn about the two main approaches: microscopic anatomy and gross, or macroscopic, anatomy."
# Check if the user's input matches a language in the dictionary
translator = Translator()
from flask import Flask, render_template, request,url_for,redirect
app=Flask(__name__)
result=text


@app.route('/',methods=['GET','POST'])
def welcome():
    return render_template('/blogpost', output=text)

@app.route('/<string:language>',methods=['GET','POST'])
def welcome2(language):
    result = translator.translate(text, dest=language).text
    return render_template('/blogpost', output=result)


@app.route('/submit',methods=['GET'])
def submit():
    selected_fruit = request.args.get('fruits')
    if selected_fruit=='english':
        language='en'
    else:
        language=languages[selected_fruit]
    return(redirect(url_for('welcome2',language=language)))


@app.route('/search',methods=['GET'])
def search():
    return render_template('/search')

if __name__=='__main__':
    app.run()