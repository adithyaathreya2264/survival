_RAMP = " .:-=+*#%@"


def line_chart(series, width=100, height=18, title=""):
    labels = list(series)
    marks = "*o+x"[: len(labels)]
    n = max(len(v) for v in series.values())
    lo = min(min(v) for v in series.values())
    hi = max(max(v) for v in series.values())
    if hi == lo:
        hi = lo + 1

    # Downsample each series to `width` buckets by averaging.
    def bucket(vals):
        out = []
        for x in range(width):
            a = x * n // width
            b = max(a + 1, (x + 1) * n // width)
            chunk = vals[a:b] or vals[-1:]
            out.append(sum(chunk) / len(chunk))
        return out

    cols = {lab: bucket(series[lab]) for lab in labels}
    canvas = [[" "] * width for _ in range(height)]
    for lab, mark in zip(labels, marks):
        for x, v in enumerate(cols[lab]):
            y = int((v - lo) / (hi - lo) * (height - 1))
            canvas[height - 1 - y][x] = mark

    out = []
    if title:
        out.append(title)
    for r, row in enumerate(canvas):
        val = hi - (hi - lo) * r / (height - 1)
        out.append(f"{val:8.1f} |" + "".join(row))
    out.append(" " * 9 + "+" + "-" * width)
    out.append(" " * 10 + f"gen 0{' ' * (width - 14)}gen {n - 1}")
    out.append(" " * 10 + "  ".join(f"{m}={lab}" for lab, m in zip(labels, marks)))
    return "\n".join(out)
