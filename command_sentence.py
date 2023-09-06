from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import cv2
import numpy as np 
import os 
from rescoring_command import find_most_similar_sentence

sinhala_words_model = tf.keras.models.load_model(r"D:\Python\python_project\sinhala_voice_api\trained_model_full")

def sinhala_word_inference(filepath):
    classes = ['0_command_spec', '0_spec', '10_command_spec', '10_spec', '11_command_spec', '11_spec', '12_spec', 
               '13_spec', '14_spec', '15_spec', '16_spec', '17_spec', '1_command_spec', 
               '1_spec', '2_command_spec', '2_spec', '3_command_spec', '3_spec', '4_command_spec', 
               '4_spec', '5_command_spec', '5_spec', '6_command_spec', '6_spec', '7_command_spec', 
               '7_spec', '8_command_spec', '8_spec', '9_command_spec', '9_spec']
    
    # classes_sinhala_words =['සැලසේ', 'පෞද්ගලික', 'ක්ෂේත්‍රයන්ට', 'ඇතිවේ', 'වෙන් කෙරේ', 'වැසි', 'විශ්වවිද්‍යාලය', 'මම', 'ගෙදර', 'හා', 'ණය', 'අධ්‍යාපනය', 'රාත්‍රියේ දී', 'පහසුකම්', 'යමි', 'මුදල්', 'ශිෂ්‍යයන්ට', 'ආර්ථික']
    image=cv2.imread(filepath)
    img_height,img_width=224,224
    image_resized= cv2.resize(image, (img_height,img_width))
    image=np.expand_dims(image_resized,axis=0)
    pred=sinhala_words_model.predict(image)
    output_class=classes[np.argmax(pred)]
    return output_class





def load_mp3_16k_mono(filename):

    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess_mp3(sample, index):
    sample = sample[0] #accessing first element of tensor
    zero_padding = tf.zeros([30000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram



def get_audio_duration(wav):
    sample_rate = 16000  
    audio_duration = len(wav) / sample_rate
    return audio_duration



def command_sentence_preprocess(file_path):
    wav = load_mp3_16k_mono(file_path)
    audio_duration = get_audio_duration(wav)
    target_word_duration = int(audio_duration)
    print(audio_duration , target_word_duration)
    sequence_length = int((audio_duration/target_word_duration) * 16000)  # Convert to number of samples
    sequence_stride = int(sequence_length / 2) 
    print(sequence_length , sequence_stride)

    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=sequence_length, sequence_stride=sequence_stride, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)


    prediction_list_before_rescoring = []

    for index , i in enumerate(audio_slices.as_numpy_iterator()):
        file_name = f"temp_spec_{index}.png"
        folder_path = r"D:\Python\python_project\sinhala_voice_api\temp_files"
        file_path_png = os.path.join(folder_path, file_name)
        plt.figure(figsize=(20, 20))
        plt.imshow(tf.transpose(i)[0])
        plt.axis('off')
        plt.savefig(file_path_png, bbox_inches='tight', pad_inches=0)
        prediction = sinhala_word_inference(file_path_png)
        prediction_list_before_rescoring.append(prediction)
        os.remove(file_path_png)
        print(prediction)

    
    most_similar_sentence = find_most_similar_sentence(prediction_list_before_rescoring)
    return most_similar_sentence