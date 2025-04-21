from pydub import AudioSegment
import os

def convert_audio_to_wav(input_path, output_path=None):
    """
    Convert an MP3 or WAV file to WAV format with 44kHz sample rate and 16-bit depth.
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Check file format
        file_extension = os.path.splitext(input_path)[1].lower()
        if file_extension not in ['.mp3', '.wav']:
            raise ValueError("Input file must be in .mp3 or .wav format")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + '.wav'

        # Load the audio file
        if file_extension == '.mp3':
            audio = AudioSegment.from_mp3(input_path)
        else:  # .wav
            audio = AudioSegment.from_wav(input_path)
        
        # Set the parameters for conversion
        audio = audio.set_frame_rate(44100)  # 44 kHz
        audio = audio.set_sample_width(2)    # 16 bits
        audio = audio.set_channels(2)

        # Export as WAV
        audio.export(output_path, format='wav')
        
        print(f"Successfully converted {input_path} to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error converting file: {str(e)}")
        raise


def check_audio_properties(file_path):
    """
    Check the audio file's properties such as sample rate, bit depth, and channels.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        
        # Get properties
        sample_rate = audio.frame_rate
        channels = audio.channels
        sample_width = audio.sample_width  # Width of one audio sample in bytes (2 bytes = 16 bits)
        bit_depth = sample_width * 8  # Convert bytes to bits

        properties = {
            "Sample Rate (Hz)": sample_rate,
            "Channels": channels,
            "Bit Depth": bit_depth,
        }

        return properties
    except Exception as e:
        print(f"Error reading audio properties: {str(e)}")
        return None