# app.py (INTROSPECTION VERSION)
   import torch
   import os # For environment variables if PyTorch needs them

   def main():
       print("--- PyTorch Introspection Start ---")

       # Basic Info
       print(f"PyTorch version: {torch.__version__}")
       print(f"CUDA available: {torch.cuda.is_available()}")

       if torch.cuda.is_available():
           print(f"CUDA version (reported by PyTorch): {torch.version.cuda}")
           print(f"Number of GPUs: {torch.cuda.device_count()}")
           current_device_id = torch.cuda.current_device()
           print(f"Current GPU ID: {current_device_id}")
           print(f"Current GPU Name: {torch.cuda.get_device_name(current_device_id)}")
           print(f"GPU Capability: {torch.cuda.get_device_capability(current_device_id)}")

           # Attempt to get default device
           try:
               default_device = torch.get_default_device() # This is the one we're interested in!
               print(f"Torch Default Device (torch.get_default_device()): {default_device}")
           except AttributeError:
               print("AttributeError: torch.get_default_device() is NOT available.")
           except Exception as e:
               print(f"Error calling torch.get_default_device(): {e}")

           # Check for MPS (Apple Silicon) - less relevant for CUDA but good for completeness
           try:
               print(f"MPS available: {torch.backends.mps.is_available()}")
               if torch.backends.mps.is_available():
                   print(f"MPS built: {torch.backends.mps.is_built()}")
           except AttributeError:
               print("torch.backends.mps not available in this PyTorch build.")

       else:
           print("CUDA not available. Running on CPU.")
           # Attempt to get default device for CPU scenario
           try:
               default_device = torch.get_default_device()
               print(f"Torch Default Device (torch.get_default_device()): {default_device}")
           except AttributeError:
               print("AttributeError: torch.get_default_device() is NOT available.")
           except Exception as e:
               print(f"Error calling torch.get_default_device(): {e}")


       # Check Flash Attention related attributes (if transformers or flash_attn is imported and being tested)
       # This part is illustrative; actual check would depend on how Flash Attn is integrated
       try:
           from torch.nn.attention import SDPBackend, ولمs_SDP_NEEDS_GRAD # Example attributes
           print("SDPBackend related attributes found (suggests newer PyTorch features for attention).")
       except ImportError:
           print("torch.nn.attention.SDPBackend not found (may indicate older PyTorch or minimal install).")


       # Check if torch.compile is available
       if hasattr(torch, 'compile'):
           print("torch.compile IS available.")
       else:
           print("torch.compile is NOT available.")

       print("--- PyTorch Introspection End ---")

   if __name__ == "__main__":
       main()