
from collections import Counter
import math
import pandas as pd


sentences = pd.read_excel(r"D:\Python\python_project\sinhala_voice_api\sentece_rescoring\Book1.xlsx" , sheet_name="Sheet2")
df = pd.read_excel(r"D:\Python\python_project\sinhala_voice_api\sentece_rescoring\word_reference.xlsx")
df['encoded'] = df['word'].apply(lambda x: (x.strip()).encode('utf-8'))

COMMAND_SENTENCES = {}



for index_sentence, i in enumerate(sentences['sentence']):
    
    reference_list = []
    for j in i.split():
        # Normalize the formatting and remove non-breaking space characters
        j_normalized = j.replace("'", "").strip().replace("\xa0", "")
        
        for index, b in enumerate(df['word']):
            # Normalize the formatting and remove non-breaking space characters
            b_normalized = b.replace("'", "").strip().replace("\xa0", "")
            
            if j_normalized == b_normalized:
                reference_list.append(df['reference'][index])

    
    
    COMMAND_SENTENCES[sentences['sentence_number'][index_sentence]] = reference_list


cleaned_dict = {}

for key, value_list in COMMAND_SENTENCES.items():
    cleaned_values = [value.replace('\xa0', '') for value in value_list]
    cleaned_dict[key] = cleaned_values

final_cleaned = {}
for key, value_list in cleaned_dict.items():
    cleaned_values = [value.replace(' ', '') for value in value_list]
    final_cleaned[key] = cleaned_values




def sentece_similarity_score(counter1, counter2):
    
    words_set1 = set(counter1.keys())
    words_set2 = set(counter2.keys())
    common_words = words_set1.intersection(words_set2)
    dot_product = sum(counter1[word] * counter2[word] for word in common_words)
    magnitude1 = math.sqrt(sum(counter1[word] ** 2 for word in words_set1))
    magnitude2 = math.sqrt(sum(counter2[word] ** 2 for word in words_set2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return similarity


def find_most_similar_sentence(prediction_list_before_rescoring):
    
    similarity_list =[]
    freq_seq1 = Counter(prediction_list_before_rescoring)
    for senetence_key in final_cleaned:
        base_counter = Counter(final_cleaned[senetence_key])
        similarity_list.append({
            "sentence" : senetence_key ,
            "similarity" :sentece_similarity_score(base_counter , freq_seq1)})
    
    most_similar_sentence = max(similarity_list, key=lambda x: x["similarity"])
    print(most_similar_sentence)
    
    return (final_cleaned[most_similar_sentence["sentence"]] , most_similar_sentence['sentence'])





