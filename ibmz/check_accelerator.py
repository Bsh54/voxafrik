"""Report the IBM Z AI acceleration environment and benchmark a VoxAfrik-sized model.

This script answers three practical questions on a LinuxONE / IBM Z machine:

  1. Which compute devices does TensorFlow see?
  2. Is the Telum on-chip AI accelerator reachable, i.e. are the zDNN runtime
     library and the NNPA instruction set present?
  3. How fast does a model shaped like the VoxAfrik classifier run here?

It only reads the environment and runs a forward pass, so it is safe to run on
any machine. On hardware without the accelerator it still prints a CPU baseline.
"""
import glob
import os
import time

import numpy as np
import tensorflow as tf


def report_devices():
    print("TensorFlow version:", tf.__version__)
    physical = tf.config.list_physical_devices()
    print("Physical devices:", [(d.device_type, d.name) for d in physical])


def report_zdnn():
    """Look for the zDNN runtime that TensorFlow and llama.cpp use to reach Telum."""
    libraries = glob.glob("/usr/**/*zdnn*", recursive=True) + glob.glob("/opt/**/*zdnn*", recursive=True)
    print("zDNN runtime libraries:", libraries[:5] if libraries else "not found")

    nnpa = False
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            nnpa = "nnpa" in handle.read().lower()
    except OSError:
        pass
    print("NNPA instruction set advertised in /proc/cpuinfo:", "yes" if nnpa else "no")


def benchmark():
    """Time a forward pass on a model with the same op families as the char CNN."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(200,)),
        tf.keras.layers.Embedding(3000, 48),
        tf.keras.layers.Conv1D(128, 4, activation="relu"),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(96, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    batch = np.random.randint(0, 3000, size=(4000, 200)).astype("int32")
    model(batch[:10], training=False).numpy()  # warm up

    start = time.perf_counter()
    for _ in range(5):
        model(batch, training=False).numpy()
    per_pass_ms = (time.perf_counter() - start) * 1000 / 5
    print("VoxAfrik-sized model: %.1f ms for 4000 messages (%.0f messages/second)" % (
        per_pass_ms, 4000 / (per_pass_ms / 1000)))


if __name__ == "__main__":
    report_devices()
    report_zdnn()
    benchmark()
