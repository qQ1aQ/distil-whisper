# app.py (ULTRA-SIMPLE DIAGNOSTIC VERSION - INTROSPECTION)
import sys
import os

print("--- PYTHON SCRIPT app.py STARTED (INTROSPECTION VERSION) ---")
print("HEY THIS IS A TEST - If you see this, the correct app.py is running!")
print(f"Python version: {sys.version}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"sys.path: {sys.path}")

try:
    print(f"--- Attempting to import and introspect torch ---")
    import torch
    print(f"Successfully imported torch.")
    print(f"_Torch version: {torch.__version__}")
    print(f"_Torch file location: {torch.__file__}")

    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available'):
        print(f"_torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"_torch.cuda.device_count(): {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                try:
                    print(f"_torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
                except Exception as e_cuda_device:
                    print(f"_Could not get CUDA device name: {e_cuda_device}")
    else:
        print("_torch.cuda or torch.cuda.is_available not found, or torch.cuda is not fully initialized.")

    print(f"\n--- Introspecting 'get_default_device' ---")
    if hasattr(torch, 'get_default_device'):
        print("SUCCESS: torch module HAS the attribute 'get_default_device' (via hasattr).")
        try:
            print("Attempting to call torch.get_default_device()...")
            default_dev = torch.get_default_device()
            print(f"SUCCESS: torch.get_default_device() call result: {default_dev}")
        except Exception as e_gd_call:
            print(f"ERROR when calling torch.get_default_device(): {e_gd_call}")
    else:
        print("FAILURE: torch module DOES NOT HAVE the attribute 'get_default_device' (via hasattr).")

    print("\nAttempting direct access to torch.get_default_device...")
    try:
        func_ref = torch.get_default_device
        print(f"SUCCESS: Direct access to torch.get_default_device succeeded. Type: {type(func_ref)}")
        # Try calling it if reference was obtained
        print("Attempting to call the directly referenced torch.get_default_device()...")
        default_dev_direct = func_ref()
        print(f"SUCCESS: Call via direct reference result: {default_dev_direct}")
    except AttributeError:
        print("FAILURE: Direct access torch.get_default_device raised AttributeError.")
    except Exception as e_direct_call:
        print(f"ERROR during direct call to get_default_device (after potential successful reference): {e_direct_call}")


    print(f"\n--- Introspecting 'compile' (another PyTorch 2.0+ feature) ---")
    if hasattr(torch, 'compile'):
        print("SUCCESS: torch module HAS the attribute 'compile' (via hasattr).")
    else:
        print("FAILURE: torch module DOES NOT HAVE the attribute 'compile' (via hasattr).")
    
    print("\nAttempting direct access to torch.compile...")
    try:
        compile_func_ref = torch.compile
        print(f"SUCCESS: Direct access to torch.compile succeeded. Type: {type(compile_func_ref)}")
    except AttributeError:
        print("FAILURE: Direct access torch.compile raised AttributeError.")
    except Exception as e_compile_direct:
        print(f"ERROR during direct access/call to compile: {e_compile_direct}")

    print(f"\n--- Checking 'get_default_device' and 'compile' in dir(torch) ---")
    torch_attributes = dir(torch)
    if 'get_default_device' in torch_attributes:
        print("INFO: 'get_default_device' IS PRESENT in dir(torch).")
    else:
        print("INFO: 'get_default_device' IS NOT PRESENT in dir(torch).")

    if 'compile' in torch_attributes:
        print("INFO: 'compile' IS PRESENT in dir(torch).")
    else:
        print("INFO: 'compile' IS NOT PRESENT in dir(torch).")
    
    # print("\n--- Sample of dir(torch) (first ~100 attributes) ---")
    # print(torch_attributes[:100])


except ImportError as e_import_torch:
    print(f"CRITICAL ERROR: Failed to import torch: {e_import_torch}")
    import traceback
    traceback.print_exc()
except Exception as e_runtime:
    print(f"CRITICAL ERROR during torch diagnostics: {e_runtime}")
    import traceback
    traceback.print_exc()

print(f"\n--- DIAGNOSTICS COMPLETE (INTROSPECTION VERSION) ---")
print(f"Exiting script for debug purposes. FastAPI app will not start.")
sys.exit(1)
