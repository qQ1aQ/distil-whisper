from transformers import pipeline
import os

def main():
    try:
        print("Attempting to load Distil Whisper model...")
        # You might need to specify a different model if you're using a custom trained one
        # Make sure this model name exists on the Hugging Face Hub or is available locally
        model_id = "distil-whisper/distil-large-v3"
        pipe = pipeline("automatic-speech-recognition", model=model_id)
        print(f"Successfully loaded model: {model_id}")
        # In a real application, you would add your inference logic here
        # For now, we'll just keep the container running
        print("Model loaded. Container is running...")
        # A simple way to keep the container alive for testing
        while True:
            pass # Or a more sophisticated way to handle incoming requests
    except Exception as e:
        print(f"Error loading model or during execution: {e}")
        # Exit with a non-zero status code to indicate an error
        exit(1)

if __name__ == "__main__":
    main()
