import os
import time
import statistics
import numpy as np
from PIL import Image
import onnxruntime as ort
import maccel

# -----------------------------
# Paths
# -----------------------------
IMAGE_ROOT = r"C:\Users\015179996\npu\imagenet_val20\imagenet_val20"
VAL_LIST   = r"C:\Users\015179996\npu\imagenet_val20.txt"
MXQ_PATH   = "simple_imagenet20_model.mxq"

NUM_IMAGES = 10   # set None for full validation

# -----------------------------
# Load validation list
# -----------------------------
samples = []
with open(VAL_LIST, "r") as f:
    for line in f:
        fname, label = line.strip().split()
        samples.append((fname, int(label)))

if NUM_IMAGES is not None:
    samples = samples[:NUM_IMAGES]
    print(f"num samples:{len(samples)}")

# -----------------------------
# Image preprocessing
# -----------------------------
def load_image(filename):
    folder = filename.split('_')[0]
    full_path = os.path.join(IMAGE_ROOT, filename)

    img = Image.open(full_path).convert("RGB")
    img = img.resize((240, 240), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0  # ToTensor()
    arr = arr.transpose(2, 0, 1)                  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)             # NCHW

    return arr.astype(np.float32)

test_images = []
labels = []

for fname, label in samples:
    test_images.append(load_image(fname))
    labels.append(label)

print(labels)

# -----------------------------
# NPU inference
# -----------------------------
npu = maccel.Accelerator(0)
model = maccel.Model(MXQ_PATH)
model.launch(npu)

# Warmup
for _ in range(3):
    model.infer([test_images[0]])

npu_times = []
npu_correct = 0

for img, gt in zip(test_images, labels):
    start = time.perf_counter()

    out = model.infer([img])[0]
    out = np.asarray(out)

    print("Output shape:", out.shape)
    print("Output dtype:", out.dtype)


    # Flatten batch dimensions safely
    if out.ndim > 1:
        out = out.reshape(out.shape[0], -1)
        logits = out[0]
    else:
        logits = out
    
    print("Output shape:", out.shape)
    print("Output dtype:", out.dtype)

    npu_times.append((time.perf_counter() - start) * 1000)
    pred = int(np.argmax(logits))

    print(f"Predicted: {pred}, Actual: {gt}")

    npu_correct += int(pred == gt)

npu_mean = statistics.mean(npu_times)
npu_acc = npu_correct / len(labels)

model.dispose()

# -----------------------------
# Results
# -----------------------------
print(f"NPU: {npu_mean:.2f} ms/image ({1000/npu_mean:.1f} img/s)")
print(f"NPU Accuracy: {npu_acc*100:.1f}%")

