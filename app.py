import_failed = False
ASR_PIPELINE = None # Initialize globally
MODEL_ID = "distil-whisper/distil-large-v3" # Define globally
TORCH_DEVICE = "cpu" # Default
TORCH_DTYPE = "torch.float32" # Default

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import io
    import soundfile as sf # To ensure audio is in the correct format/rate for the model
    import numpy as np # For audio data manipulation
except ImportError:
    import_failed = True
    print("CRITICAL IMPORT ERROR: One or more essential Python packages (FastAPI, PyTorch, Transformers, soundfile, numpy) are not installed.")
    print("Please ensure your requirements.txt includes these and is correctly processed in the Dockerfile.")
    # If critical imports fail, the FastAPI app can't be defined or run.
    # We might raise here, or let the script fail when 'app = FastAPI()' is hit.
    # For now, the print is a strong indicator.

# --- Global Model Configuration & Loading ---
# This section will run once when the Uvicorn server starts and imports this module.

print("Attempting to initialize application and load ASR model...")

if not import_failed:
    try:
        TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        TORCH_DTYPE = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability(TORCH_DEVICE)[0] >= 7 else torch.float32 # float16 only for Ampere+

        print(f"Targeting device: {TORCH_DEVICE} with dtype: {TORCH_DTYPE}")

        print(f"Loading processor for {MODEL_ID}...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("Processor loaded.")

        print(f"Loading model {MODEL_ID}...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,      # Load on CPU first
            use_safetensors=True,
            attn_implementation="flash_attention_2"  # Correct way to specify Flash Attention 2
        )
        model.to(TORCH_DEVICE) # Move to target device
        print(f"Model {MODEL_ID} successfully moved to {TORCH_DEVICE}.")

        ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=TORCH_DEVICE,
            # For distil-large-v3, the pipeline should handle long-form well by default.
            # Chunking parameters can be added if specific chunked behavior is desired:
            # chunk_length_s=25,
            # batch_size=4  # Adjust batch_size based on GPU memory
        )
        print(f"ASR pipeline for {MODEL_ID} created successfully on {TORCH_DEVICE}.")
        print("Model loading complete.")

    except Exception as e:
        print(f"CRITICAL ERROR DURING MODEL LOADING: {e}")
        import traceback
        traceback.print_exc()
        ASR_PIPELINE = None # Ensure pipeline is None if loading failed
else:
    print("Skipping model loading due to initial import errors.")

# --- FastAPI Application Definition ---
# This part will only be effective if imports were successful.
if not import_failed:
    app = FastAPI()

    @app.get("/")
    async def root():
        return {
            "message": "Distil-Whisper ASR API",
            "model_id": MODEL_ID,
            "device": TORCH_DEVICE,
            "model_status": "Loaded and ready" if ASR_PIPELINE else "Failed to load or not initialized"
        }

    @app.post("/transcribe/")
    async def transcribe_audio(audio_file: UploadFile = File(...)):
        if ASR_PIPELINE is None:
            print("Transcription request failed: Model not available.")
            raise HTTPException(status_code=503, detail="Model not loaded or loading failed. Cannot process requests.")

        if not audio_file.content_type.startswith("audio/"):
            print(f"Transcription request failed: Invalid file type {audio_file.content_type}.")
            raise HTTPException(status_code=400, detail=f"Invalid file type: {audio_file.content_type}. Please upload an audio file.")

        try:
            print(f"Received file: {audio_file.filename}, content_type: {audio_file.content_type}")
            audio_bytes = await audio_file.read()
            audio_io = io.BytesIO(audio_bytes)

            # Read audio data and sampling rate
            # Using dtype='float32' as it's a common format transformers work well with internally.
            data, samplerate = sf.read(audio_io, dtype='float32', always_2d=False)
            print(f"Audio decoded with soundfile. Original samplerate: {samplerate}, Shape: {data.shape}")

            # Whisper models expect mono audio. If stereo, convert to mono.
            # soundfile's always_2d=False should give 1D array for mono.
            if data.ndim > 1 and data.shape[-1] > 1: # Check if stereo (e.g., shape (samples, 2))
                print("Audio is stereo, converting to mono by averaging channels.")
                data = np.mean(data, axis=-1)


            # The pipeline's feature_extractor will handle resampling to the model's required 16kHz.
            # The pipeline expects a dictionary with "raw" (numpy array) and "sampling_rate".
            input_audio = {"raw": data, "sampling_rate": samplerate}

            print("Sending audio to ASR pipeline for transcription...")
            result = ASR_PIPELINE(input_audio)
            
            transcription_text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()

            print(f"Transcription successful for {audio_file.filename}. Text: {transcription_text[:100]}...") # Log first 100 chars
            return JSONResponse(content={"filename": audio_file.filename, "transcription": transcription_text})

        except sf.LibsndfileError as e:
            print(f"Soundfile error processing audio '{audio_file.filename}': {e}")
            raise HTTPException(status_code=400, detail=f"Error processing audio file with soundfile: {e}. Ensure it's a valid audio format (e.g., WAV, FLAC, MP3).")
        except Exception as e:
            print(f"Error during transcription for '{audio_file.filename}': {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {str(e)}")

else:
    # If imports failed, we can't define 'app'.
    # Uvicorn would fail to start. This print is for container logs if that happens.
    print("FastAPI app cannot be initialized due to critical import errors.")

# To run this app (typically done by Uvicorn in the Docker CMD):
# uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
# The --workers 1 flag is important for PyTorch models on GPU to avoid multiprocessing issues
# unless your model/pipeline is specifically designed for multi-worker GPU handling.
