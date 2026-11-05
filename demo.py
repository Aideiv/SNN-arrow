"""Demonstration: train the 2-layer LIF SNN encoder and compare vs baseline."""

import torch

from hdc_snn import (
    SNNHDCEncoder,
    baseline_encoder,
    compare_encoders,
    make_place_sequences,
    seed_everything,
)

DIM = 16384
SEED = 0


def main() -> None:
    seed_everything(SEED)

    # ------------------------------------------------------------------ #
    # 1. Deterministic synthetic place sequences
    # ------------------------------------------------------------------ #
    events, labels = make_place_sequences(
        num_places=8, samples_per_place=16, num_events=200, seed=SEED
    )
    print(f"Data: events {tuple(events.shape)}  labels {tuple(labels.shape)}")
    print(f"       places {labels.unique().tolist()}")

    # ------------------------------------------------------------------ #
    # 2. Model (exactly 2 LIF layers + fixed random HDC projection)
    # ------------------------------------------------------------------ #
    model = SNNHDCEncoder(dim=DIM, seed=SEED)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    # Check the strict interface: single [N,4] and batched [B,N,4] both work.
    code_single = model.encode(events[0])          # [N, 4] -> [D]
    code_batch = model.encode(events[:3])          # [B, N, 4] -> [B, D]
    assert code_single.shape == (DIM,)
    assert code_batch.shape == (3, DIM)
    print(f"encode([N,4]) -> {tuple(code_single.shape)}, "
          f"encode([B,N,4]) -> {tuple(code_batch.shape)}")

    # ------------------------------------------------------------------ #
    # 3. Train
    # ------------------------------------------------------------------ #
    print("\nTraining...")
    history = model.train(
        events,
        labels,
        epochs=100,
        batch_size=32,
        lr=1e-3,
        temperature=0.5,
        lambda_sparse=1e-4,
        lambda_bin=1e-2,
        seed=SEED,
    )
    print(f"Loss: {history[0]:.4f} -> {history[-1]:.4f}")

    # Determinism check: re-encoding the same data gives identical codes.
    model.eval()
    with torch.no_grad():
        c1 = model.encode(events)
        c2 = model.encode(events)
    assert torch.equal(c1, c2), "encoder is not deterministic"
    print("Determinism: identical re-encode verified")

    # ------------------------------------------------------------------ #
    # 4. Side-by-side comparison vs time-bin + random projection baseline
    # ------------------------------------------------------------------ #
    baseline = baseline_encoder(dim=DIM, seed=SEED)
    snn_res, base_res = compare_encoders(
        events, labels, snn_encode=model.encode, baseline_encode=baseline.encode
    )

    print("\n" + "=" * 66)
    for r in (snn_res, base_res):
        print(f"{r.name}")
        print(f"  recall@1                             : {r.recall_at_1:.3f}")
        print(f"  mean sim to correct prototype        : {r.mean_similarity_to_correct:.3f}")
        print(f"  mean sim to OTHER prototypes (~0.5)  : {r.mean_inter_place_similarity:.3f}")
        print(f"  mean output sparsity (active bits)   : {r.mean_sparsity:.3f}")
    print("=" * 66)

    # Requirement: same place -> high similarity, different place -> ~0.5.
    # The trained SNN must separate same vs different places.
    assert snn_res.mean_similarity_to_correct > snn_res.mean_inter_place_similarity, (
        "SNN should assign higher similarity to the correct prototype"
    )
    assert abs(snn_res.mean_inter_place_similarity - 0.5) < 0.15, (
        "cross-place similarity should stay near 0.5"
    )


if __name__ == "__main__":
    main()