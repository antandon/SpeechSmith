# import myspsolution as mysp
from conversion import convert_audio_to_wav,check_audio_properties
mysp=__import__("my-voice-analysis")
                     
p="Sample" # Audio File title
c=r"C:\\Users\\Sheela Sai kumar\\Documents\\UPSkilling\\ML\\Experiments\\Speaksmith\\speech.wav" # Path to the Audio_File directory (Python 3.7)

enchanced_audio = convert_audio_to_wav(c)
print(enchanced_audio)
properties = check_audio_properties(enchanced_audio)

if properties:
    print("Audio Properties:")
    for key, value in properties.items():
        print(f"{key}: {value}")

mysp.myspgend(p,enchanced_audio)
# rate of speech
mysp.myspsr(p,c)

