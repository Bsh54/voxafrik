"""Run IBM Granite on the Telum II accelerator to explain why a message was flagged.

The real-time classifier (see backend/) decides NORMAL / ABUSIVE / HATE in well under
a millisecond on the s390x CPU. This module is the optional second layer: once a post
is flagged, a small IBM Granite model writes a short, human-readable reason for the
moderator.

Granite runs here through llama.cpp built with the GGML zDNN backend. With
"n_gpu_layers=-1" llama.cpp offloads its matrix multiplications to the Telum II
on-chip AI accelerator (NNPA) via zDNN, exactly as a CUDA build would offload to a
GPU. When llama.cpp starts it prints a "ggml_zdnn" initialisation line, which is how
we confirmed the work really lands on the accelerator and not on the CPU.

Honest note: the model we had access to is "granite-4.0-h-350m-base", a 350M base
(not instruction-tuned) checkpoint. It proves the accelerator path end to end, but its
free-text reasoning is too weak to ship to moderators, so the live product keeps the
fast ensemble and treats this layer as research. A larger instruction-tuned Granite on
the same code path would be the production upgrade.

Usage:
    pip install -r ibmz/requirements.txt       # llama-cpp-python built against zDNN
    export GRANITE_GGUF=/path/to/granite-4.0-h-350m-base-be.f16.gguf
    python ibmz/granite_telum.py "your tribe should leave our country"
"""
import os
import sys
import time

from llama_cpp import Llama

MODEL_PATH = os.environ.get("GRANITE_GGUF", "granite-4.0-h-350m-base-be.f16.gguf")

# A few labelled examples steer the base model toward one short, on-topic sentence.
FEW_SHOT = """Give one short reason why a message is harmful or safe.
Message: "all women are stupid and should stay in the kitchen"
Reason: Demeans women based on their gender.
Message: "those immigrants are criminals, throw them out"
Reason: Xenophobic attack on immigrants as a group.
Message: "god bless you my friend, have a great day"
Reason: Not harmful, a friendly greeting.
Message: "{message}"
Reason:"""


def load_model():
    """Load Granite with every layer offloaded to the Telum accelerator (zDNN)."""
    return Llama(model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=-1, verbose=True)


def explain(llm, message):
    prompt = FEW_SHOT.format(message=message)
    start = time.time()
    output = llm(prompt, max_tokens=24, temperature=0.2, stop=["\n", "Message:"])
    seconds = time.time() - start
    return output["choices"][0]["text"].strip(), seconds


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "your tribe is a disease and should leave our country"
    model = load_model()
    reason, seconds = explain(model, message)
    print("message:", message)
    print("reason :", reason)
    print("generated in %.1fs on the accelerator path" % seconds)
