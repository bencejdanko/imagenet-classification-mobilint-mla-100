import os
import time
import statistics
import zipfile
import numpy as np
from PIL import Image
import onnxruntime as ort
import maccel
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Hugging Face Settings
# -----------------------------
MODEL_REPO = "bdanko/imagenetsub20"
DATA_REPO = "bdanko/imagenetsubset20"
DATA_DIR = "data"

# -----------------------------
# Download Models
# -----------------------------
print(f"Downloading models from {MODEL_REPO}...")
MXQ_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="imagenetsub20resnet10-5-calibrated.mxq")
ONNX_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="imagenetsub20resnet10-5.onnx")
ONNX_DATA_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="imagenetsub20resnet10-5.onnx.data")

# -----------------------------
# Download and Prepare Dataset
# -----------------------------
print(f"Downloading and preparing dataset from {DATA_REPO}...")
os.makedirs(DATA_DIR, exist_ok=True)

# Download val list
VAL_LIST = hf_hub_download(repo_id=DATA_REPO, filename="imagenet_val20.txt", repo_type="dataset")

# Download and extract images zip
val_zip_path = hf_hub_download(repo_id=DATA_REPO, filename="imagenet_val20.zip", repo_type="dataset")
IMAGE_ROOT = os.path.join(DATA_DIR, "imagenet_val20")

if not os.path.exists(IMAGE_ROOT):
    print(f"Extracting {val_zip_path} to {DATA_DIR}...")
    with zipfile.ZipFile(val_zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

# Adjust for potential nested folder in zip (original script used .../imagenet_val20/imagenet_val20)
if os.path.exists(os.path.join(IMAGE_ROOT, "imagenet_val20")):
    IMAGE_ROOT = os.path.join(IMAGE_ROOT, "imagenet_val20")


NUM_IMAGES = None   # set None for full validation

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
    full_path = os.path.join(IMAGE_ROOT, filename)

    img = Image.open(full_path).convert("RGB")
    # Matches transforms.Resize((240, 240)) in augmentation.py
    img = img.resize((240, 240), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0  # ToTensor()
    arr = arr.transpose(2, 0, 1)                  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)             # NCHW

    # Normalization (Matches transforms.Normalize in augmentation.py)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    arr = (arr - mean) / std

    return arr.astype(np.float32)

test_images = []
labels = []

for fname, label in samples:
    test_images.append(load_image(fname))
    labels.append(label)

# print(labels)

# -----------------------------
# CPU Inference (Onnx runtime)
# -----------------------------
cpu_session = ort.InferenceSession(ONNX_PATH)
input_name = cpu_session.get_inputs()[0].name

for _ in range(3):  # warmup
    cpu_session.run(None, {input_name: test_images[0]})

cpu_times = []
cpu_preds = []

for img in test_images:
    start = time.perf_counter()

    outputs = cpu_session.run(None, {input_name: img})

    elapsed = (time.perf_counter() - start) * 1000
    cpu_times.append(elapsed)

    logits = outputs[0]

    # Handle batch vs non-batch safely
    if logits.ndim == 2:
        logits = logits[0]

    pred_class = int(np.argmax(logits))
    cpu_preds.append(pred_class)

cpu_mean = statistics.mean(cpu_times)

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

    # Flatten batch dimensions safely
    if out.ndim > 1:
        out = out.reshape(out.shape[0], -1)
        logits = out[0]
    else:
        logits = out

    npu_times.append((time.perf_counter() - start) * 1000)
    pred = int(np.argmax(logits))

    npu_correct += int(pred == gt)

npu_mean = statistics.mean(npu_times)
npu_acc = npu_correct / len(labels)

model.dispose()

# -----------------------------
# Results
# -----------------------------
cpu_correct = sum(1 for p, g in zip(cpu_preds, labels) if p == g)
cpu_acc = cpu_correct / len(labels)

print(f"\nFinal Results on {len(labels)} images:")
print(f"CPU Performance: {cpu_mean:.2f} ms/image ({1000/cpu_mean:.1f} img/sec), Accuracy: {cpu_acc*100:.2f}%")
print(f"NPU Performance: {npu_mean:.2f} ms/image ({1000/npu_mean:.1f} img/sec), Accuracy: {npu_acc*100:.2f}%")
print(f"Speedup: {cpu_mean/npu_mean:.1f}x")
