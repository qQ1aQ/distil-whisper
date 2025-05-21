import_failed = False
ASR_PIPELINE = None
MODEL_ID = "distil-whisper/distil-large-v3"
TORCH_DEVICE = "cpu"
TORCH_DTYPE = "torch.float32"

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    import torch
    import sys # For printing sys.path for more detailed debugging
    # +++ DEBUG LINES START +++
    print(f"DEBUG: torch.__version__ = {torch.__version__}")
    print(f"DEBUG: torch.__file__ = {torch.__file__}")
    # You can uncomment the line below for very detailed path debugging if needed, but it can be lengthy.
    # print(f"DEBUG: sys.path = {sys.path}")
    # +++ DEBUG LINES END +++
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import io
    import soundfile as sf
    import numpy as np
except ImportError as e_import: # Give the exception a different name
    import_failed = True
    print(f"CRITICAL IMPORT ERROR: {e_import}") # Print the actual import error
    print("Please ensure your requirements.txt includes these and is correctly processed in the Dockerfile.")
    # If critical imports fail, the FastAPI app can't be defined or run.
    # We might raise here, or let the script fail when 'app = FastAPI()' is hit.
    # For now, the print is a strong indicator.

# --- Global Model Configuration & Loading ---
print("Attempting to initialize application and load ASR model...")

if not import_failed:
    try:
        TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Ensure float16 is used only on GPUs that support it well (Ampere+)
        if torch.cuda.is_available() and torch.cuda.get_device_capability(TORCH_DEVICE)[0] >= 7:
            TORCH_DTYPE = torch.float16
        else:
            TORCH_DTYPE = torch.float32
            if torch.cuda.is_available(): # If on GPU but not Ampere+, warn about float32
                 print(f"Warning: GPU {torch.cuda.get_device_name(TORCH_DEVICE)} capability "
                       f"{torch.cuda.get_device_capability(TORCH_DEVICE)} may not optimally support float16. Using float32.")


        print(f"Targeting device: {TORCH_DEVICE} with dtype: {TORCH_DTYPE}")

        print(f"Loading processor for {MODEL_ID}...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("Processor loaded.")

        print(f"Loading model {MODEL_ID}...")
        # This is where the error occurs if torch.get_default_device is missing
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2"
        )
        model.to(TORCH_DEVICE)
        print(f"Model {MODEL_ID} successfully moved to {TORCH_DEVICE}.")

        ASR_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=TORCH_DEVICE,
        )
        print(f"ASR pipeline for {MODEL_ID} created successfully on {TORCH_DEVICE}.")
        print("Model loading complete.")

    except Exception as e_load: # Give the exception a different name
        print(f"CRITICAL ERROR DURING MODEL LOADING OR SETUP: {e_load}")
        import traceback
        traceback.print_exc()
        ASR_PIPELINE = None
else:
    print("Skipping model loading due to initial import errors.")

# --- FastAPI Application Definition ---
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

            data, samplerate = sf.read(audio_io, dtype='float32', always_2d=False)
            print(f"Audio decoded with soundfile. Original samplerate: {samplerate}, Shape: {data.shape}")

            if data.ndim > 1 and data.shape[-1] > 1:
                print("Audio is stereo, converting to mono by averaging channels.")
                data = np.mean(data, axis=-1)

            input_audio = {"raw": data, "sampling_rate": samplerate}

            print("Sending audio to ASR pipeline for transcription...")
            result = ASR_PIPELINE(input_audio)
            
            transcription_text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()

            print(f"Transcription successful for {audio_file.filename}. Text: {transcription_text[:100]}...")
            return JSONResponse(content={"filename": audio_file.filename, "transcription": transcription_text})

        except sf.LibsndfileError as e_sf:
            print(f"Soundfile error processing audio '{audio_file.filename}': {e_sf}")
            raise HTTPException(status_code=400, detail=f"Error processing audio file with soundfile: {e_sf}. Ensure it's a valid audio format.")
        except Exception as e_transcribe:
            print(f"Error during transcription for '{audio_file.filename}': {e_transcribe}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {str(e_transcribe)}")
else:
    print("FastAPI app cannot be initialized due to critical import errors.")
