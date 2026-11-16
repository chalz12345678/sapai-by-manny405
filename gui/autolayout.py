"""
Automatic screen layout: finds where the pets are from the game's own
graphics, so nobody has to click positions or drag boxes.

Every pet in battle shows its health in a red heart, with its attack in a
dark stone just left of it, at the same height for every pet and with the
same spacing between pets. The hearts are found by colour, the stones
confirm them (a red hat or background is not a heart), and a regular grid
is fitted through them, so missing hearts (a misdrawn one, an empty slot)
are filled in. Every other box is placed from the grid, in proportion to
the spacing between pets (S), as measured on the real game:

    attack stone   0.49 S left of the heart, same height
    pet picture    0.84 S wide, centred 0.70 S above the heart; hats and
                   held food sit in the top 45%, so the box used to tell
                   pets apart is the lower part, below any hat
    level number   1.43 S above the heart; 0.36 S right of the middle of
                   the stats on the opponent's side, 0.12 S left of it on
                   yours (just the number, which a hat over the badge does
                   not cover)
    front pets     the attack-and-health pairs of the two front pets sit
                   0.70 S either side of the middle of the screen

Fractions of the frame, like a layout set up by hand.
"""

import numpy as np

### Proportions, in units of the spacing between pets (S)
STONE_DX = -0.49
PAIR_DX = STONE_DX / 2  # the middle of an attack-and-health pair
FRONT_GAP = 0.70  # pair middle of a front pet to the middle of the screen
PET_DY = -0.70  # centre of the whole pet picture above the heart
PET_SIZE = 0.84
BODY_TOP = 0.45  # hats and perks: the top 45% of the picture
NUMBER_W, NUMBER_H = 0.32, 0.30
### The level badge is drawn above the attack stone on your side and above
###   the heart on the opponent's: the number's offset from the middle of
###   the attack-and-health pair, for each side
LEVEL_DX = {"player": -0.12, "opponent": 0.36}
LEVEL_DY = -1.43
LEVEL_W, LEVEL_H = 0.20, 0.30
### A perk sits in different places on different pets (on the head, on the
###   back), so this box only covers where it usually is: the upper middle
PERK_DY, PERK_W, PERK_H = -0.98, 0.40, 0.36


def red_mask(img):
    a = img.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 180) & (g < 80) & (b < 90)


def blobs(mask, max_pixels=200_000):
    """Connected regions: list of (x0, y0, x1, y1, n pixels)"""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    ys, xs = np.nonzero(mask)
    if len(ys) > max_pixels:
        return out
    for y, x in zip(ys, xs):
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        x0 = x1 = x
        y0 = y1 = y
        n = 0
        while stack:
            cy, cx = stack.pop()
            n += 1
            x0, x1, y0, y1 = min(x0, cx), max(x1, cx), min(y0, cy), max(y1, cy)
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append((x0, y0, x1 + 1, y1 + 1, n))
    return out


def _shrink(img, max_w=960):
    """Work at most max_w wide (positions come back as fractions)"""
    h, w = img.shape[:2]
    step = max(1, int(np.ceil(w / max_w)))
    return img[::step, ::step], step


def _has_stone(a, cx, cy, hw, hh):
    """A dark grey attack stone about 1.4 heart widths left of a heart"""
    sx = int(cx - 1.4 * hw)
    patch = a[
        int(cy - hh / 4) : int(cy + hh / 4), max(0, sx - hw // 3) : max(1, sx + hw // 3)
    ]
    if patch.size == 0:
        return False
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    grey = (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (r < 140)
    return grey.mean() >= 0.2


def _hearts_on_row(small, red, y, w):
    """Hearts on the row at height y: runs of columns with red in them (the
    white number splits a heart, so small gaps are bridged)"""
    band = int(0.016 * w)
    rows = red[max(0, y - band) : y + band]
    cols = rows.sum(0) >= 2
    a = small.astype(np.int16)
    out, x, n = [], 0, len(cols)
    bridge = int(0.009 * w)
    while x < n:
        if not cols[x]:
            x += 1
            continue
        start, gap = x, 0
        while x < n and gap <= bridge:
            gap = 0 if cols[x] else gap + 1
            x += 1
        end = x - gap
        hw = end - start
        if 0.018 * w <= hw <= 0.05 * w:
            ys = np.flatnonzero(red[max(0, y - band) : y + band, start:end].any(1))
            hh = ys[-1] - ys[0] + 1
            cy = max(0, y - band) + (ys[0] + ys[-1]) / 2
            cx = (start + end) / 2
            if _has_stone(a, cx, cy, hw, hh):
                out.append((cx, cy, hw, hh))
    return out


def hearts(img):
    """
    The hearts on the pets' row, as (x, y, width, height) in pixels of img.
    Tries each height where something red is, and keeps the one with the
    most hearts that each have an attack stone beside them (a row of red
    hats has no stones)
    """
    small, step = _shrink(img)
    w = small.shape[1]
    red = red_mask(small)
    heights = sorted({int((b[1] + b[3]) / 2) for b in blobs(red) if b[4] >= 12})
    best, tried = [], set()
    for y in heights:
        if any(abs(y - t) < 0.004 * w for t in tried):
            continue
        tried.add(y)
        found = _hearts_on_row(small, red, y, w)
        if len(found) > len(best):
            best = found
    return [(x * step, y * step, bw * step, bh * step) for x, y, bw, bh in best]


### Pets are this many heart widths apart (89 px for a 28-30 px heart)
SPACING_PER_HEART = 3.1


def spacing(row):
    """
    Spacing between neighbouring pets: first guessed from the hearts' width,
    then measured from the hearts that are a whole number of places apart
    (the gap between the two front pets is not, and is left out)
    """
    guess = SPACING_PER_HEART * float(np.median([bw for _, _, bw, _ in row]))
    xs = sorted(x for x, _, _, _ in row)
    steps = []
    for i, a in enumerate(xs):
        for b in xs[i + 1 :]:
            n = round((b - a) / guess)
            if 1 <= n <= 4 and abs((b - a) / n - guess) < 0.12 * guess:
                steps.append((b - a) / n)
    return float(np.median(steps)) if steps else guess


def battle_layout(img):
    """
    A Battle Lab layout for a battle frame (a frame with pets on both sides,
    such as the start of a battle). Returns (layout, report); raises
    ValueError when it can't find enough pets.
    """
    h, w = img.shape[:2]
    row = sorted(hearts(img))
    if len(row) < 3:
        raise ValueError(
            "Couldn't find the pets: pick a frame from the start of a battle, "
            "with pets on both sides and their attack and health showing"
        )
    s = spacing(row)
    hy = float(np.median([y for _, y, _, _ in row]))
    mid = w / 2
    ### Which slot each heart is: pairs sit FRONT_GAP from the middle, then
    ###   one spacing apart; refine the gap from the hearts found
    gap = FRONT_GAP * s
    for _ in range(3):
        offsets = []
        for x, _, _, _ in row:
            d = abs(x + PAIR_DX * s - mid)
            k = max(0, round((d - gap) / s))
            offsets.append(d - k * s)
        gap = float(np.median(offsets))
    slots = {"player": [], "opponent": []}
    for x, _, _, _ in row:
        side = "player" if x + PAIR_DX * s < mid else "opponent"
        k = max(0, round((abs(x + PAIR_DX * s - mid) - gap) / s))
        if k < 5:
            slots[side].append(k)
    body_h = PET_SIZE * (1 - BODY_TOP)
    body_cy = hy + (PET_DY + PET_SIZE / 2 - body_h / 2) * s
    layout = {"mirror": False}
    for side, sign in (("player", -1), ("opponent", 1)):
        layout[side] = [
            [(mid + sign * (gap + k * s)) / w, body_cy / h] for k in range(5)
        ]
    layout["slot"] = [PET_SIZE * s / w, body_h * s / h]

    def part(dx, dy, bw, bh):
        ### Offsets from the slot point (pair middle, body centre)
        return [dx * s / w, (dy * s + hy - body_cy) / h, bw * s / w, bh * s / h]

    layout["attack"] = part(PAIR_DX, 0, NUMBER_W, NUMBER_H)
    layout["health"] = part(-PAIR_DX, 0, NUMBER_W, NUMBER_H)
    layout["level"] = part(LEVEL_DX["player"], LEVEL_DY, LEVEL_W, LEVEL_H)
    layout["level_opponent"] = part(LEVEL_DX["opponent"], LEVEL_DY, LEVEL_W, LEVEL_H)
    ### The opponent's pets face the other way: flip their pictures, so a
    ###   pet learned on one side is known on the other
    layout["flip_opponent"] = True
    layout["auto"] = True
    layout["perk"] = part(0, PERK_DY, PERK_W, PERK_H)
    report = {
        "hearts": len(row),
        "spacing_px": round(s, 1),
        "found": {side: sorted(set(v)) for side, v in slots.items()},
    }
    return layout, report
