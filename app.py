# app.py (ULTRA-SIMPLE DIAGNOSTIC VERSION)
import sys
import os

print("--- PYTHON SCRIPT app.py STARTED ---")
print("HEY THIS IS A TEST - If you see this, the correct app.py is running!")
print(f"Python version: {sys.version}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"sys.path: {sys.path}") # Shows where Python looks for modules

try:
    print(f"Attempting to import torch...")
    import torch
    print(f"Successfully imported torch.")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch file location: {torch.__file__}")

    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available'):
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                # Attempt to get device name, handle potential errors if CUDA isn't fully initialized
                try:
                    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
                except Exception as e_cuda_device:
                    print(f"Could not get CUDA device name: {e_cuda_device}")
    else:
        print("torch.cuda or torch.cuda.is_available not found, or torch.cuda is not fully initialized.")

    if hasattr(torch, 'get_default_device'):
        print("torch module HAS the attribute 'get_default_device'")
        try:
            default_dev = torch.get_default_device()
            print(f"torch.get_default_device() call attempt, result: {default_dev}")
        except Exception as e_gd_call:
            print(f"Error when calling torch.get_default_device(): {e_gd_call}")
    else:
        print("torch module DOES NOT HAVE the attribute 'get_default_device'")

except ImportError as e_import_torch:
    print(f"CRITICAL ERROR: Failed to import torch: {e_import_torch}")
    import traceback
    traceback.print_exc()
except Exception as e_runtime:
    print(f"CRITICAL ERROR during torch diagnostics: {e_runtime}")
    import traceback
    traceback.print_exc()

print(f"--- DIAGNOSTICS COMPLETE ---")
print(f"Exiting script for debug purposes. FastAPI app will not start.")
sys.exit(1) # Exit with an error code so Uvicorn doesn't try to run a non-existent app object

#
# === THE REST OF YOUR FastAPI APP IS EFFECTIVELY COMMENTED OUT FOR THIS TEST ===
#
# import_failed = False
# ASR_PIPELINE = None
# ... (rest of your original app.py) ...
#
