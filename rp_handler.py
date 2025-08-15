import runpod
import whisper
import tempfile
import requests
import os

def handler(event):
    input_data = event.get('input', {})
    audio_url = input_data.get('audio')
    model_name = input_data.get('model', 'base')
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

    # Load Whisper model
    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        os.unlink(temp_audio_path)
        return {"error": f"Failed to load model '{model_name}': {str(e)}"}

    # Transcribe
    try:
        result = model.transcribe(temp_audio_path, language=language)
    except Exception as e:
        os.unlink(temp_audio_path)
        return {"error": f"Transcription failed: {str(e)}"}
    finally:
        os.unlink(temp_audio_path)

    return {"text": result.get('text', ''), "language": result.get('language', language)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
