from fastapi import FastAPI ,Path , File, UploadFile , Form
# from typing import Optional
# from pydantic import BaseModel
import os
from sinhala_words_model import word_preprocess
from sinhala_sentence import sentence_preprocess
from sinhala_command_words import command_preprocess
from command_sentence import  command_sentence_preprocess
from intention_recognition import intention_sentence
app = FastAPI()

@app.post("/predict/sinhala-word")
async def predict_sinhala_word(file: UploadFile = File):
    file_name = file.filename
    #change the file path when hosting !!
    file_path = rf"D:\Python\python_project\sinhala_voice_api\temp_files\{file_name}"
    with open(file_path , "wb") as f:
        content = await file.read()
        f.write(content)
        response = word_preprocess(rf'{file_path}')
        f.close()
        os.remove(file_path)
        return (response)

@app.post("/predict/sinhala-command")
async def predict_sinhala_word(file: UploadFile = File):
    file_name = file.filename
    #change the file path when hosting !!
    file_path = rf"D:\Python\python_project\sinhala_voice_api\temp_files\{file_name}"
    with open(file_path , "wb") as f:
        content = await file.read()
        f.write(content)
        response = command_preprocess(rf'{file_path}')
        f.close()
        os.remove(file_path)
        return (response)

@app.post("/predict/sinhala-word-sentence")
async def predict_sinhala_word(file: UploadFile = File):
    file_name = file.filename
    #change the file path when hosting !!
    file_path = rf"D:\Python\python_project\sinhala_voice_api\temp_files\{file_name}"
    with open(file_path , "wb") as f:
        content = await file.read()
        f.write(content)
        response = sentence_preprocess(rf'{file_path}')
        f.close()
        os.remove(file_path)
        return (response)

@app.post("/predict/sinhala-command-sentence")
async def predict_sinhala_word(file: UploadFile = File):
    file_name = file.filename
    #change the file path when hosting !!
    file_path = rf"D:\Python\python_project\sinhala_voice_api\temp_files\{file_name}"
    with open(file_path , "wb") as f:
        content = await file.read()
        f.write(content)
        sentence , sentence_number = command_sentence_preprocess(rf'{file_path}')
        intention_response = intention_sentence(sentence_number)
        f.close()
        os.remove(file_path)
        return {
            "sentence" : sentence,
            "intension_response" : intention_response 
        }