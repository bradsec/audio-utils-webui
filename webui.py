import os
import json
import shutil
import gradio as gr
from pathlib import Path
import demucs.separate
import zipfile
import librosa
import soundfile
from slicer2 import Slicer
import whisper
from whisper.utils import get_writer

def banner_info():
    try:
        # Load the code sources information from the JSON file
        with open('repos.json', 'r') as json_file:
            code_sources = json.load(json_file)
    except Exception as e:
        return str(e)

    # Prepare the banner and source information
    banner = f'''
                  _ _                               
   __ _ _   _  __| (_) ___                          
  / _` | | | |/ _` | |/ _ \                         
 | (_| | |_| | (_| | | (_) |                        
  \__,_|\__,_|\__,_|_|\___/           _           _ 
  _   _| |_(_) |___     __      _____| |__  _   _(_)
 | | | | __| | / __|____\ \ /\ / / _ \ '_ \| | | | |
 | |_| | |_| | \__ \_____\ V  V /  __/ |_) | |_| | |
  \__,_|\__|_|_|___/      \_/\_/ \___|_.__/ \__,_|_|                                                                                 
    '''
    source_info = "\nUtilities and Code Used in WebUI:\n"
    sources = ''.join(
        f'\n{source["name"]: <18} [{source["github"]: <25}]'
        for source in code_sources
    )
    
    print(f'{banner} {source_info} {sources}\n')


def slice_audio(input_audio, threshold=-40, min_length=5000, min_interval=300, hop_size=10, max_sil_kept=500):

    # Ensure the input audio file is provided
    if not input_audio:
        return None, "Input audio file is required."
    
    print("Audio Slicing Commenced...")
    
    # Show some terminal output for command being run.
    print(f"Executing with parameters: threshold={threshold}, "
          f"min_length={min_length}, min_interval={min_interval}, hop_size={hop_size}, max_sil_kept={max_sil_kept}")
    
    output_dir = init_output_dir()
    
    try:
        # Load the audio file
        audio, sr = librosa.load(input_audio, sr=None, mono=False)
    except Exception as e:
        return None, str(e)
    
    try:
        # Initialize the Slicer and slice the audio
        slicer = Slicer(
            sr=sr,
            threshold=threshold,
            min_length=min_length,
            min_interval=min_interval,
            hop_size=hop_size,
            max_sil_kept=max_sil_kept
        )
        chunks = slicer.slice(audio)
    except Exception as e:
        return None, str(e)
    
    # Prepare the output paths
    output_paths = []
    try:
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T
            output_filename = f'sliced_{i}.wav'
            output_path = os.path.join(output_dir, output_filename)
            soundfile.write(output_path, chunk, sr)
            output_paths.append(output_path)
    except Exception as e:
        return None, str(e)
    
    # Prepare the zip file
    zip_filename = os.path.join(output_dir, "sliced_audio_chunks.zip")
    try:
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file in output_paths:
                zipf.write(file, os.path.basename(file))
    except Exception as e:
        return None, None, str(e)
    
    print("Audio Slicing Complete.")

    return zip_filename, output_paths, None


# Clean and setup output file directory
def init_output_dir(dir_path="temp_output"):
    output_dir = Path(dir_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return dir_path

def pretty_format_json(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

def separate_audio(input_audio, model, two_stems, audio_type):
    # Ensure the input audio file is provided
    if not input_audio:
        return [None] * 6, "Input audio file is required."
    
    print("Demucs Audio Seperation Commenced...")
    
    output_dir = init_output_dir()

    args = [
        "-v",
        "-o", str(output_dir),
        "-n", model, 
        input_audio,
    ]

    # Set the audio type and extension
    file_extension = {"mp3": ".mp3", "wav": ".wav", "flac": ".flac"}.get(audio_type, ".wav")
    if audio_type == "mp3":
        args.extend(["--mp3", "--mp3-preset", "2"])
    elif audio_type == "flac":
        args.append("--flac")
    
    # Set the stem options
    if two_stems:
        args.extend(["--two-stems", "vocals"])
        stem_names = ["vocals", "no_vocals"]
    elif model == "htdemucs_6s":
        stem_names = ["vocals", "other", "bass", "drums", "piano", "guitar"]
    else:
        stem_names = ["vocals", "other", "bass", "drums"]

    # Execute the demucs separation
    try:
        print("Executing command: demucs " + ' '.join(args))
        demucs.separate.main(args)
    except Exception as e:
        return [None] * 6, str(e)
    
    # Prepare the output paths
    output_paths = []
    for stem in stem_names:
        stem_path = Path(output_dir) / model / Path(input_audio).stem / (stem + file_extension)
        print(f'stem_path: {stem_path}')
        if stem_path.exists():
            output_paths.append(str(stem_path))
        else:
            output_paths.append(None)
    
    # Ensure the output paths list has 6 elements
    output_paths.extend([None] * (6 - len(output_paths)))

    print("Demucs Audio Seperation Completed.")

    return output_paths, None


def whisper_transcription(input_audio, whisper_model):

    # Ensure the input audio file is provided
    if not input_audio:
        return [None] * 6, "Input audio file is required."
    
    print("Whisper Audio Translation Commenced...")
    
    print(f'Using the {whisper_model} model...')
    
    model = whisper.load_model(whisper_model)
    result = model.transcribe(input_audio, language="english", verbose=True)
    output_dir = init_output_dir()

    # Get the filename without extension
    filename_without_extension = os.path.splitext(os.path.basename(input_audio))[0]

    txt_file_path = os.path.join(output_dir, f"{filename_without_extension}.txt")
    json_file_path = os.path.join(output_dir, f"{filename_without_extension}.json")
    vtt_file_path = os.path.join(output_dir, f"{filename_without_extension}.vtt")
    tsv_file_path = os.path.join(output_dir, f"{filename_without_extension}.tsv")
    srt_file_path = os.path.join(output_dir, f"{filename_without_extension}.srt")

    options = {
        'max_line_width': None,
        'max_line_count': None,
        'highlight_words': False
    }

    text_blob = result['text']

    txt_writer = get_writer("txt", output_dir)
    txt_writer(result, input_audio, options)

    json_writer = get_writer("json",  output_dir)
    json_writer(result, input_audio, options)

    vtt_writer = get_writer("vtt", output_dir)
    vtt_writer(result, input_audio, options)

    tsv_writer = get_writer("tsv", output_dir)
    tsv_writer(result, input_audio, options)

    srt_writer = get_writer("srt", output_dir)
    srt_writer(result, input_audio, options)

    pretty_format_json(output_dir)

    print("Whisper Audio Translation Completed.")

    # Return the paths of the output files
    return text_blob, txt_file_path, json_file_path, vtt_file_path, tsv_file_path, srt_file_path


def main():
    banner_info()
    init_output_dir()
    title = "audio-utils-webui"
    description = """
    # audio-utils-webui
    ## A Gradio WebUI for Audio Utilities.
    """

    repo_link = """
    [audio-utils-webui](https://github.com/bradsec/audio-utils-webui)  
    """

    demucs_info = """
    **htdemucs**: First version of Hybrid Transformer Demucs.   
    Trained on MusDB + 800 songs. Default model.  
    **htdemucs_ft**: Fine-tuned version of htdemucs.   
    Separation will take 4 times more time but might be a bit better. Same training set as htdemucs.  
    **htdemucs_6s**: 6 sources version of htdemucs, with piano and guitar being added as sources.   
    Note that the piano source is not working great at the moment.  
    **hdemucs_mmi**: Hybrid Demucs v3, retrained on MusDB + 800 songs.  
    **mdx**: Trained only on MusDB HQ, winning model on track A at the MDX challenge.   
    **mdx_extra**: Trained with extra training data (including MusDB test set).   
    Ranked 2nd on the track B of the MDX challenge.  
    **mdx_q, mdx_extra_q**: Quantized version of the previous models.   
    Smaller download and storage but quality can be slightly worse.  
    **SIG**: Where SIG is a single model from the model zoo.    
       
    Source: [facebookresearch/demucs](https://github.com/facebookresearch/demucs).  
    """

    audio_slicer_info = """
    **sr**: Sampling rate of the input audio.  
    **db_threshold**:The RMS threshold presented in dB.   
    Areas where all RMS values are below this threshold will be regarded as silence.   
    Increase this value if your audio is noisy. Defaults to -40.  
    **min_length**: The minimum length required for each sliced audio clip, presented in milliseconds.  
    **min_interval**: The minimum length for a silence part to be sliced, presented in milliseconds.   
    Set this value smaller if your audio contains only short breaks.   
    The smaller this value is, the more sliced audio clips this script is likely to generate.   
    Note that this value must be smaller than min_length and larger than hop_size. Defaults to 300.  
    **hop_size**: Length of each RMS frame, presented in milliseconds.   
    Increasing this value will increase the precision of slicing, but will slow down the process.  
    **max_silence_kept**: The maximum silence length kept around the sliced audio, presented in milliseconds.   
    Adjust this value according to your needs.   
    Note that setting this value does not mean that silence parts in the sliced audio have exactly the given length.   
    The algorithm will search for the best position to slice, as described above.  
        
    Source: [openvpi/audio-slicer](https://github.com/openvpi/audio-slicer).  
    """

    whisper_info = """
    ## Available models and languages

    There are five model sizes, four with English-only versions, offering speed and accuracy tradeoffs. Below are the names of the available models and their approximate memory requirements and relative speed. 


    |  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
    |:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
    |  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
    |  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
    | small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
    | medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
    | large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |

    The `.en` models for English-only applications tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.
          
    Source: [github.com/openai/whisper](https://github.com/openai/whisper).
    """


    with gr.Blocks(title=title) as audio_webui:
        
        gr.Markdown(description)

        with gr.Tab("Demucs Music Seperation"):
            gr.Markdown("""
            ## Music Source Separation with Demucs
            Demucs is a state-of-the-art music source separation model, currently capable of separating drums, bass, and vocals from the rest of the accompaniment.             
            """)
            
            with gr.Row():
                with gr.Column():
                    # Inputs
                    input_audio = gr.Audio(type="filepath", label="Input Audio")
                    gr.Markdown(" ")
                    two_stems = gr.Checkbox(label="Two Stems", info="Check for two audio output files only - vocals and other instrumentals.", value=False)
                    with gr.Accordion("About Models", open=False):
                        gr.Markdown(demucs_info)
                    demucs_model = gr.Radio(["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"], label="Select Model", info="Previously unused models will have to download upon first use. Check terminal status.", value="htdemucs")
                    audio_type = gr.Radio(["mp3", "wav", "flac"], label="Output Audio Type", value="mp3")
                    process_audio_button = gr.Button("Process Audio", variant="primary")
                
                with gr.Column():
                    gr.Markdown("""
                    *Only outputs applicable to model chosen will be populated.*
                    """)
                    # Outputs
                    output_audio_vocals = gr.Audio(type="filepath", label="Output Audio - Vocals")
                    output_audio_other = gr.Audio(type="filepath", label="Output Audio - Other")
                    output_audio_bass = gr.Audio(type="filepath", label="Output Audio - Bass")
                    output_audio_drums = gr.Audio(type="filepath", label="Output Audio - Drums")
                    output_audio_piano = gr.Audio(type="filepath", label="Output Audio - Piano")
                    output_audio_guitar = gr.Audio(type="filepath", label="Output Audio - Guitar")
            
            def on_process_audio(*args):
                output_paths, err = separate_audio(*args)
                if err:
                    raise gr.Error(err)
                return output_paths

            process_audio_button.click(on_process_audio, 
                                    inputs=[input_audio, demucs_model, two_stems, audio_type], 
                                    outputs=[output_audio_vocals, output_audio_other, output_audio_bass, output_audio_drums, output_audio_piano, output_audio_guitar])
            
        with gr.Tab("Audio Slicer"):
            gr.Markdown("""
            ## Audio Slicing
            Slice audio files into chunks based on silence detection and set parameters.
            """)
            
            with gr.Row():
                    with gr.Column():
                        # Inputs
                        input_audio = gr.Audio(type="filepath", label="Input Audio")
                        with gr.Accordion("About Parameters", open=False):
                            gr.Markdown(audio_slicer_info)
                        threshold = gr.Slider(minimum=-60, maximum=0, value=-40, label="Threshold (dB)")
                        min_length = gr.Slider(minimum=1000, maximum=20000, value=5000, step=100, label="Min Length (ms)")
                        min_interval = gr.Slider(minimum=100, maximum=1000, value=300, step=10, label="Min Interval (ms)")
                        hop_size = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Hop Size (ms)")
                        max_sil_kept = gr.Slider(minimum=0, maximum=5000, value=500, step=100, label="Max Silence Kept (ms)")
                        slice_audio_button = gr.Button("Slice Audio", variant="primary")
                    
                    with gr.Column():
                        # Outputs
                        output_audio_chunks_zip = gr.File(label="Sliced Audio Chunks (Zip)", type="file")
                        output_audio_chunks_list = gr.File(label="Sliced Audio Chunks List", type="file")
                    
            def on_slice_audio(*args):
                zip_filename, output_paths, err = slice_audio(*args)
                if err:
                    raise gr.Error(err)
                return zip_filename, output_paths

            slice_audio_button.click(on_slice_audio, 
                                    inputs=[input_audio, threshold, min_length, min_interval, hop_size, max_sil_kept], 
                                    outputs=[output_audio_chunks_zip, output_audio_chunks_list])
            
        with gr.Tab("Whisper Translation"):
            gr.Markdown("""
            ## Audio Translation using OpenAI's Whisper
            Convert audio speech to text. WebUI currently only supports english translation.
            """)
            
            with gr.Row():
                with gr.Column():
                    # Inputs
                    input_audio = gr.Audio(type="filepath", label="Input Audio")
                    with gr.Accordion("About Models", open=False):
                        gr.Markdown(whisper_info)
                    whisper_model = gr.Radio(["tiny", "base", "small", "medium", "large-v2"], label="Select Model", info="Previously unused models will have to download upon first use. Check terminal status.", value="base")
                    translate_audio_button = gr.Button("Translate Audio", variant="primary")
                
                with gr.Column():
                    # Outputs
                    output_text_blob = gr.Textbox(label="Audio Transcript", lines=5, show_copy_button=True)
                    output_text_transcript = gr.File(label="Audio Transcript (.txt file format)", type="file")
                    output_json_transcript = gr.File(label="Audio Transcript (.json file format)", type="file")
                    output_vtt_transcript = gr.File(label="Audio Transcript (.vtt file format)", type="file")
                    output_tsv_transcript = gr.File(label="Audio Transcript (.tsv file format)", type="file")
                    output_srt_transcript = gr.File(label="Audio Transcript (.srt file format)", type="file")

            translate_audio_button.click(
                whisper_transcription, 
                inputs=[input_audio, whisper_model], 
                outputs=[output_text_blob, output_text_transcript, output_json_transcript, output_vtt_transcript, output_tsv_transcript, output_srt_transcript]
            )

        gr.Markdown(repo_link)
        
    # Launch the Gradio web server
    audio_webui.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()