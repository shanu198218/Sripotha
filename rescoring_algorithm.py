from collections import Counter
import math
import pandas as pd 


BASE_SENTENCES = {
    "sentence_1_Base": ["16_spec", "17_spec", "6_spec"],
    "sentence_2_Base": ["4_spec", "14_spec", "12_spec"],
    "sentence_3_Base": ["9_spec", "1_spec", "3_spec", "11_spec", "7_spec", "13_spec"],
    "sentence_4_Base": ["10_spec", "15_spec", "8_spec", "2_spec", "5_spec", "0_spec"]
}



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
    for senetence_key in BASE_SENTENCES:
        base_counter = Counter(BASE_SENTENCES[senetence_key])
        similarity_list.append({
            "sentence" : senetence_key ,
            "similarity" :sentece_similarity_score(base_counter , freq_seq1)})
    
    most_similar_sentence = max(similarity_list, key=lambda x: x["similarity"])
    print(most_similar_sentence)
    
    return BASE_SENTENCES[most_similar_sentence["sentence"]]


