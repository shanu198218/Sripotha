
import pandas as pd
from googletrans import Translator, constants
import spacy
import re


translator = Translator()
nlp = spacy.load("en_core_web_sm")
sentences = pd.read_excel(r"D:\Python\python_project\sinhala_voice_api\sentece_rescoring\Book1.xlsx" , sheet_name="Sheet2")
df = pd.read_excel(r"D:\Python\python_project\sinhala_voice_api\sentece_rescoring\word_reference.xlsx")
df['encoded'] = df['word'].apply(lambda x: (x.strip()).encode('utf-8'))

# using spacy and googletrans return action in the given sentence
def action_recognition(sentence):

  doc = nlp(sentence)
  subject = None
  action = None

  for token in doc:
      if "subj" in token.dep_:
          subject = token.text
      elif "ROOT" in token.dep_:
          action = token.text
  return action


# using spacy and googletrans return subject in the given sentence
def subject_recognition(sentence):

  match = re.search(r"'([^']+)'", sentence)
  if match:
      word_within_quotes = match.group(1)
      index = sentence.index(word_within_quotes)
      return word_within_quotes
  else:
      return None 

# using spacy and googletrans return properties in the given sentence
def colour_property(sentence):
  
  sentence = sentence.lower()
  color_words = ["red", "blue", "green", "yellow"]
  words = sentence.split()
  matching_colors = [word for word in words if word.strip("'") in color_words]
  if matching_colors:
    return matching_colors[0]
  else:
    return None 

# using spacy and googletrans return starting index and ending index in the given sentence
def subject_index(sentence , word):

  starting_index = sentence.find(word)
  ending_index = starting_index + len(word) - 1
  return starting_index , ending_index 
 







def intention_sentence(sentence_number):
    data = []
    for index , i in enumerate(sentences['sentence_number']):
        if sentence_number == i:
            translation = translator.translate(sentences['sentence'][index], src="si",dest="en")
            action = action_recognition(translation.text)
            subject = subject_recognition(sentences['sentence'][index])
            starting_index , ending_index  = subject_index(sentences['sentence'][index] , subject)
            subject_property = colour_property(translation.text)
            # print(f"{translation.origin} ({translation.src}) --> {(translation.text)} ({action}) ({subject}) ({subject_property})")
            data.append({
                    'Original Sentence': translation.origin,
                    'Translated Sentence': translation.text,
                    'Action': action,
                    'Subject': subject,
                    "Starting_index" : starting_index ,
                    "Ending_index" : ending_index+1 , 
                    'Subject Property': subject_property.replace("'", '') if subject_property else None 
                })
    return data
            
            

           

