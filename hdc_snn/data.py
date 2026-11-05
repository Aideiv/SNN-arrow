"""Deterministic synthetic event-stream generator for place recognition."""

from __future__ import annotations

import torch


def make_place_sequences(
    num_places: int = 8,
    samples_per_place: int = 16,
    num_events: int = 200,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``[total, num_events, 4]`` event streams and 1-D place labels.

    Each place owns a few random "landmark" centroids (seeded). Events from a
    place are scattered around its landmarks, so sequences from the same place
    are statistically similar while different places are mostly distinct.
    Coordinates are normalized to ``[0, 1]``; timestamps to ``[0, 1]``;
    polarity is +/-1.
    """
    torch.manual_seed(seed)
    landmarks_per_place = 4
    center = torch.rand(num_places, landmarks_per_place, 2)  # in [0,1]^2
    spread = 0.03

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for place in range(num_places):
        for _ in range(samples_per_place):
            lm = torch.randint(0, landmarks_per_place, (num_events,))
            c = center[place][lm]
            pos = c + spread * torch.randn(num_events, 2)
            pos = pos.clamp(0.0, 1.0)
            # Non-uniform timestamps so temporal order carries information.
            t, _ = torch.sort(torch.rand(num_events))
            p = torch.where(torch.rand(num_events) < 0.5, -1.0, 1.0).to(
                torch.float32
            )
            ev = torch.cat([pos, p.unsqueeze(1), t.unsqueeze(1)], dim=1)
            xs.append(ev)
            ys.append(place)

    events = torch.stack(xs)
    labels = torch.tensor(ys, dtype=torch.long)
    return events, labels


def make_place_prototypes(events: torch.Tensor, labels: torch.Tensor, dim: int) -> torch.Tensor:
    """Reference place prototypes from a time-bin + random projection baseline.

    Used only to report an out-of-the-box (untrained) baseline similarity.
    """
    from .encode_utils import TimeBinProjectionEncoder

    enc = TimeBinProjectionEncoder(dim=dim, seed=0)
    codes = enc.encode(events)
    places = labels.unique()
    protos = torch.stack(
        [codes[labels == p].float().mean(dim=0) for p in places]
    )
    # Make identical anchor copies so same-place similarity starts near 1.
    return protos