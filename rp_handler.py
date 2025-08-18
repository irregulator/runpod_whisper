
import runpod
import tempfile
import requests
import os
from transformers import pipeline

def handler(event):
    input_data = event.get('input', {})
    audio_url = input_data.get('audio')
    model_name = input_data.get('model', 'Sandiago21/whisper-large-v2-greek')
    language = input_data.get('language', None)

    if not audio_url:
        return {"error": "Missing 'audio' parameter."}

    # Download audio file
    try:
        response = requests.get(audio_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(response.content)
            temp_audio_path = temp_audio.name
    except Exception as e:
        return {"error": f"Failed to download audio: {str(e)}"}

    # Load Huggingface Whisper pipeline
    try:
        asr = pipeline("automatic-speech-recognition", model=model_name, device=0)
    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return {"error": f"Failed to load model '{model_name}': {str(e)}"}

    # Transcribe
    try:
        result = asr(temp_audio_path, return_timestamps=True)
    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return {"error": f"Transcription failed: {str(e)}"}
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

    return {"text": result.get('text', ''), "language": language}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
