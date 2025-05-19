import_failed = False
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import io
    import soundfile as sf # To ensure audio is in the correct format/rate for the model
except ImportError:
    import_failed = True # Keep a flag for basic print if server can't even start
    print("One or more essential Python packages (FastAPI, PyTorch, Transformers, soundfile) are not installed.")
    print("Please ensure your requirements.txt is correctly processed in the Dockerfile.")
    # Exit here if critical imports fail, as the FastAPI app won't run
    raise

# --- Global Model Configuration ---
# This section will run once when the Uvicorn server starts and imports this module.
MODEL_ID = "distil-whisper/distil-large-v3"
TORCH_DEVICE = None
TORCH_DTYPE = None
ASR_PIPELINE = None

print("Starting application and loading model...")

if not import_failed:
    try:
        TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Using device: {TORCH_DEVICE} with dtype: {TORCH_DTYPE}")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            use_flash_attention_2=True  # Enable Flash Attention 2
        )
        model.to(TORCH_DEVICE)

        # If Flash Attention 2 is not supported or fails, you might want a fallback to BetterTransformer
        # e.g. by catching an error during model loading with use_flash_attention_2=True
        # and then trying:
        # model = AutoModelForSpeechSeq2Seq.from_pretrained(...) # without flash attention
        # model.to(TORCH_DEVICE)
        # model = model.to_bettertransformer()

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=TORCH_DEVICE,
            # For long audio files, the pipeline automatically handles chunking
            # with distil-large-v3 supporting sequential long-form transcription.
            # You can add chunk_length_s and batch_size for 'chunked' long-form if preferred:
            # chunk_length_s=25, # Optimal for distil-large-v3 if using 'chunked'
            # batch_size=16,
        )
        print(f"Model {MODEL_ID} loaded successfully on {TORCH_DEVICE}.")
    except Exception as e:
        print(f"Error loading the model or creating the pipeline: {e}")
        # If the model fails to load, ASR_PIPELINE will remain None.
        # The API endpoints will then return an error.
        ASR_PIPELINE = None # Ensure it's None if loading failed
else:
    print("Skipping model loading due to import errors.")


# --- FastAPI Application ---
app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Distil-Whisper ASR API",
        "model_id": MODEL_ID,
        "device": TORCH_DEVICE,
        "status": "Model loaded" if ASR_PIPELINE else "Model loading failed or not loaded"
    }

@app.post("/transcribe/")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    if ASR_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded or loading failed. Cannot process requests.")

    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        # Read audio file contents
        audio_bytes = await audio_file.read()
        
        # Use soundfile to read the audio bytes and get the raw audio array and sampling rate
        # This helps ensure the audio is in a format the pipeline expects.
        # The pipeline expects a NumPy array or a path to a file.
        # It also expects the audio to be mono and at the model's sampling rate (16kHz for Whisper).
        
        # Create a file-like object for soundfile
        audio_io = io.BytesIO(audio_bytes)
        
        # Read audio data and sampling rate
        # soundfile might raise an error if the format is not recognized or corrupted
        data, samplerate = sf.read(audio_io, dtype='float32') # Read as float32

        # The pipeline's feature_extractor will handle resampling if needed,
        # but it's good practice to be aware of the input format.
        # Whisper models expect mono audio. If stereo, convert to mono (e.g., by averaging channels)
        if data.ndim > 1 and data.shape[1] > 1: # Check if stereo
            data = data.mean(axis=1) # Convert to mono by averaging channels
        
        print(f"Received audio: {audio_file.filename}, content_type: {audio_file.content_type}, original samplerate: {samplerate}")

        # Perform transcription
        # The pipeline can accept the raw audio data (NumPy array) directly.
        # It will also handle resampling to the model's required sampling rate (16kHz for Whisper)
        # result = ASR_PIPELINE(audio_bytes) # This also works for many formats
        result = ASR_PIPELINE({"raw": data, "sampling_rate": samplerate})


        print(f"Transcription successful for {audio_file.filename}")
        return JSONResponse(content={"filename": audio_file.filename, "transcription": result["text"]})

    except sf.LibsndfileError as e:
        print(f"Soundfile error processing audio: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing audio file with soundfile: {e}. Ensure it's a valid audio format.")
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {str(e)}")

# Note: Uvicorn will run this app. For example:
# uvicorn app:app --host 0.0.0.0 --port 8000
# This is handled by the CMD in your Dockerfile.
