from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import cv2
import numpy as np 
import os 

sinhala_words_model = tf.keras.models.load_model(r"D:\Python\python_project\sinhala_voice_api\trained_model")




def sinhala_word_inference(filepath):
    # classes = ['0_command_spec', '0_spec', '10_command_spec', '10_spec', '11_command_spec', '11_spec', '12_spec', 
    #            '13_spec', '14_spec', '15_spec', '16_spec', '17_spec', '1_command_spec', 
    #            '1_spec', '2_command_spec', '2_spec', '3_command_spec', '3_spec', '4_command_spec', 
    #            '4_spec', '5_command_spec', '5_spec', '6_command_spec', '6_spec', '7_command_spec', 
    #            '7_spec', '8_command_spec', '8_spec', '9_command_spec', '9_spec']
    
    # classes = ['0_command_spec', '10_command_spec',
    #            '11_command_spec', '1_command_spec',
    #            '2_command_spec', '3_command_spec', 
    #            '4_command_spec', '5_command_spec', 
    #            '6_command_spec', '7_command_spec', 
    #            '8_command_spec', '9_command_spec']
    
    classes = ['0_spec', '10_spec', '11_spec', 
               '12_spec', '13_spec', '14_spec', 
               '15_spec', '16_spec', '17_spec', 
               '1_spec', '2_spec', '3_spec', '4_spec', 
               '5_spec', '6_spec', '7_spec', '8_spec', '9_spec']
    
    image=cv2.imread(filepath)
    img_height,img_width=224,224
    image_resized= cv2.resize(image, (img_height,img_width))
    image=np.expand_dims(image_resized,axis=0)
    pred=sinhala_words_model.predict(image)
    print(pred[0])
    output_class=classes[np.argmax(pred)]
    print(output_class)
    return output_class




def load_wav_16k_mono(filename):

    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

    return wav

def word_preprocess(file_path):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:30000]
    zero_padding = tf.zeros([30000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    file_name = "temp_spec.png"
    folder_path = r"D:\Python\python_project\sinhala_voice_api\temp_files"
    file_path_png = os.path.join(folder_path, file_name)
    plt.figure(figsize=(20, 20))
    # plt.plot(tf.transpose(spectrogram)[0])
    plt.imshow(tf.transpose(spectrogram)[0])
    plt.axis('off')
    plt.savefig(file_path_png, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    prediction = sinhala_word_inference(file_path_png)
   
    return prediction 


