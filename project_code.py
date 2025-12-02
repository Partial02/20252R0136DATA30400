# ------------------------
# Dummy baseline for Kaggle submission
# Generates random multi-label predictions
# ------------------------
import os
import csv
import random
from tqdm import tqdm

# --- Paths ---
TEST_DIR = "Amazon_products/test"  # modify if needed
TEST_CORPUS_PATH = os.path.join(TEST_DIR, "test_corpus.txt")  # product_id \t text
SUBMISSION_PATH = "submission.csv"  # output file

# --- Constants ---
NUM_CLASSES = 531  # total number of classes (0–530)
MIN_LABELS = 1     # minimum number of labels per sample
MAX_LABELS = 3     # maximum number of labels per sample

# --- Load test corpus ---
def load_corpus(path):
    """Load test corpus into {pid: text} dictionary."""
    pid2text = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                pid, text = parts
                pid2text[pid] = text
    return pid2text

pid2text_test = load_corpus(TEST_CORPUS_PATH)
pid_list_test = list(pid2text_test.keys())

# --- Generate random predictions ---
all_pids, all_labels = [], []
for pid in tqdm(pid_list_test, desc="Generating dummy predictions"):
    n_labels = random.randint(MIN_LABELS, MAX_LABELS)
    labels = random.sample(range(NUM_CLASSES), n_labels)
    labels = sorted(labels)
    all_pids.append(pid)
    all_labels.append(labels)

# --- Save submission file ---
with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["pid", "labels"])
    for pid, labels in zip(all_pids, all_labels):
        writer.writerow([pid, ",".join(map(str, labels))])

print(f"Dummy submission file saved to: {SUBMISSION_PATH}")
print(f"Total samples: {len(all_pids)}, Classes per sample: {MIN_LABELS}-{MAX_LABELS}")
