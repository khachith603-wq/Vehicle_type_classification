import sys
try:
    from transformers import pipeline
    print("Transformers imported successfully")
except ImportError:
    print("Transformers not found")
