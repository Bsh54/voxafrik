# Running VoxAfrik on IBM Z

This folder holds the parts of VoxAfrik that are specific to IBM LinuxONE and the
Telum II processor. Nothing here is required to run the classifier or the console;
it documents and reproduces how we used the hardware.

## The two layers and where they run

VoxAfrik has two layers, and each one runs on the part of the chip it suits.

1. The real-time classifier (in `backend/`) is a small word-embedding and
   character-CNN ensemble. It scores a message in well under a millisecond on the
   s390x CPU, which is what makes continuous screening of a live stream possible.
   A small model on a fast general-purpose core is the right tool for this job.

2. The generative layer (this folder) is an IBM Granite language model that, once a
   message is flagged, writes a short natural-language reason for the moderator.
   Text generation is matrix-multiplication heavy, which is exactly the workload the
   Telum II on-chip AI accelerator is built for, so this layer runs on the
   accelerator.

## How Granite reaches the Telum accelerator

The accelerator is reached through zDNN, IBM's deep-learning runtime for Telum. We
run Granite with `llama.cpp` built against its GGML zDNN backend. Passing
`n_gpu_layers=-1` tells llama.cpp to offload every layer to the accelerator through
zDNN, the same way a CUDA build offloads to a GPU. At start-up llama.cpp prints a
`ggml_zdnn` initialisation line; seeing that line is how we confirmed the compute
really lands on the accelerator and not on the CPU.

## Files

| File | What it does |
|---|---|
| `check_accelerator.py` | Prints the TensorFlow devices, looks for the zDNN runtime and the NNPA instruction set, and benchmarks a VoxAfrik-sized model. |
| `granite_telum.py` | Loads IBM Granite through llama.cpp with full accelerator offload and generates a one-sentence reason for a flagged message. |
| `requirements.txt` | The extra `llama-cpp-python` dependency for the Granite layer. |

## Reproduce

```bash
# 1. Inspect the acceleration environment and get a CPU-or-accelerator benchmark.
python ibmz/check_accelerator.py

# 2. Install a zDNN-enabled llama.cpp and point at a Granite GGUF checkpoint.
pip install -r ibmz/requirements.txt
export GRANITE_GGUF=/path/to/granite-4.0-h-350m-base-be.f16.gguf
python ibmz/granite_telum.py "your tribe should leave our country"
```

## Honest result

The accelerator path works end to end: Granite loads, offloads to Telum through zDNN,
and returns text. The checkpoint we had, `granite-4.0-h-350m-base`, is a 350M base
model that is not instruction-tuned, so its free-text reasoning is not reliable enough
to put in front of a moderator. We therefore ship the fast ensemble for the live
decision and keep this generative layer as a demonstrated, reproducible research path.
The production upgrade is a larger instruction-tuned Granite on this same code path.
