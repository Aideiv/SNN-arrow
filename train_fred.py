"""Train and compare the upgraded SNN encoder vs a learned CNN baseline on FRED.

Objective: scene (recording-sequence) recognition from temporal windows of
FRED event frames. Both encoders produce 16384-bit binary hypervectors and are
trained with an identical prototype contrastive loss, then evaluated with
prototype-based retrieval metrics.
"""

from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader

from hdc_snn.conv_encoder import CNNBaselineHDC, SpikingConvHDC, encode_batch, fit_retrieval
from hdc_snn.encode_utils import (
    mean_inter_place_similarity,
    mean_similarity_to_correct,
    prototypes_from_codes,
    recall_at_1,
    sparsity,
)
from hdc_snn.fred_data import make_fred_split

ROOT = os.path.join(os.path.dirname(__file__), "data", "fred")
TRAIN_IDS = ["36", "108", "67", "35"]
TEST_IDS = ["21", "111", "115", "65"]

DIM = 16384
WINDOW = 8
SIZE = 64
EPOCHS = 6
BATCH = 64
DEVICE = "cpu"


def report(name: str, codes: torch.Tensor, labels: torch.Tensor) -> None:
    protos = prototypes_from_codes(codes, labels)
    r1 = recall_at_1(codes, labels, protos)
    sim_correct = mean_similarity_to_correct(codes, labels, protos)
    sim_inter = mean_inter_place_similarity(codes, labels, protos)
    sp = float(sparsity(codes).mean().item())
    print(f"\n{name}")
    print(f"  recall@1                          : {r1:.3f}")
    print(f"  mean sim to correct prototype     : {sim_correct:.3f}")
    print(f"  mean sim to OTHER prototypes (~.5): {sim_inter:.3f}")
    print(f"  mean output sparsity (active bits): {sp:.3f}")


def main() -> None:
    torch.manual_seed(0)
    split = make_fred_split(
        ROOT, TRAIN_IDS, TEST_IDS, window=WINDOW, stride=4, size=SIZE, train_frac=0.7
    )
    print(f"FRED train windows: {len(split.train)}  test windows: {len(split.test)}")

    train_loader = DataLoader(split.train, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(split.test, batch_size=BATCH, shuffle=False)

    # ---- SNN encoder -------------------------------------------------
    snn = SpikingConvHDC(
        dim=DIM, in_channels=1, channels=(24, 48), pool_size=2,
        window=WINDOW, beta=0.9, threshold=0.5, surrogate_scale=10.0, seed=0,
    )
    snn.set_num_classes(len(TRAIN_IDS))
    t0 = time.time()
    hist = fit_retrieval(
        snn, train_loader, epochs=EPOCHS, lr=2e-3, temperature=0.1,
        lambda_sparse=1e-4, device=DEVICE, seed=0,
    )
    print(f"\nSNN train time {time.time()-t0:.1f}s  loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    snn_codes, snn_labels = encode_batch(snn, test_loader, DEVICE)
    report("Spiking conv HDC (upgraded SNN)", snn_codes, snn_labels)

    # ---- learned CNN baseline ---------------------------------------
    cnn = CNNBaselineHDC(
        dim=DIM, in_channels=1, channels=(24, 48), pool_size=2, seed=0,
    )
    cnn.set_num_classes(len(TRAIN_IDS))
    t0 = time.time()
    hist = fit_retrieval(
        cnn, train_loader, epochs=EPOCHS, lr=2e-3, temperature=0.1,
        lambda_sparse=0.0, device=DEVICE, seed=0,
    )
    print(f"\nCNN baseline train time {time.time()-t0:.1f}s  loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    cnn_codes, cnn_labels = encode_batch(cnn, test_loader, DEVICE)
    report("Learned CNN baseline (non-SNN)", cnn_codes, cnn_labels)


if __name__ == "__main__":
    main()