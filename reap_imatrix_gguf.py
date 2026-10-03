#!/usr/bin/env python3
"""
Reap (remove) MoE experts from an *imatrix* GGUF, keeping ONLY the experts listed
per layer in a selection JSON file.

This is the companion to reap_gguf.py. When you prune routed experts out of a
model, llama-quantize fails with:
    "imatrix size 327680 is different from tensor size 204800 for blk.*.ffn_down_exps.weight"
because the importance arrays in the imatrix still describe the OLD expert count.
This script slices those importance arrays along the expert axis so they match the
reaped model, letting llama-quantize proceed.

How it works (lossless, byte-slice — no requant):
  - An imatrix is a GGUF whose tensors are the per-tensor importance profiles:
        <tensor>.weight.in_sum2   (importance sums)
        <tensor>.weight.counts    (sample counts)
  - For routed-expert tensors the LAST element dimension is the expert axis
    (e.g. ffn_down_exps.weight.in_sum2 = [640, n_experts]; .counts = [1, n_experts]).
    The router (ffn_gate_inp.*) has no expert axis and is left untouched.
  - GGUF stores the last dim as the slowest-varying (contiguous) block, so each
    expert is a contiguous run of bytes. We keep only the selected experts by
    gathering their byte-blocks — exact same dtype, no precision loss.

Everything model-specific (expert count, tensor shapes, dtypes) is read from the
source imatrix at runtime — nothing is hardcoded. The imatrix has NO
general.architecture key, so expert_count is inferred from the `.counts` tensors
(their trailing dim is 1 for non-expert tensors and n_experts for expert ones).

Selection JSON format (same as reap_gguf.py):
      {"0": [0, 1, 2, 4, ...], "1": [...], ..., "47": [...]}
  Or a single flat list applied identically to every block:
      [0, 1, 2, 4, ...]

Usage:
    python reap_imatrix_gguf.py <selection.json> <source.imatrix.gguf> <output.imatrix.gguf> [--verbose]
"""

import sys
import json
import struct
from pathlib import Path

from gguf import GGUFReader, GGUFValueType, GGML_QUANT_SIZES


# Substrings that identify MoE routed-expert / router importance tensors.
# Only tensors whose name contains one of these AND whose last dim equals the
# expert count are sliced. This prevents accidentally slicing unrelated
# importance arrays that merely share the same numeric size (e.g. attention
# k/v with key length == n_kv_heads*head_dim == expert_count in full-attention
# layers). Importance tensors are named like "<tensor>.weight.in_sum2"/".counts",
# so we check the underlying weight name portion.
MOE_NAME_MARKERS = ("_exps", "ffn_gate_inp", "expert")

# Substrings that mark attention importance — never slice these.
ATTN_EXCLUDE = ("attn_", ".qkv", "_q.", "_k.", "_v.", "_o.")


def fmt_size(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def prod_int(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def tensor_byte_size_from_shape(shape, ttype):
    """Exact on-data byte size of a tensor given its element shape & dtype.

    Independent of inter-tensor alignment padding (which is added AFTER a tensor).
    Works for plain dtypes (F32/F16 -> block_size 1) and quantized types.
    """
    ttype = int(ttype)
    if ttype not in GGML_QUANT_SIZES:
        raise ValueError(f"Unknown dtype {ttype} for byte-size computation")
    block_size, type_size = GGML_QUANT_SIZES[ttype]
    n_elems = prod_int(shape)
    if n_elems % block_size != 0:
        raise ValueError(f"Tensor element count {n_elems} not divisible by "
                         f"{block_size}-element blocks for dtype {ttype}")
    return (n_elems // block_size) * type_size


def write_kv_value(fout, kv_type, value):
    if kv_type == GGUFValueType.STRING:
        vb = value.encode("utf-8")
        fout.write(struct.pack("<Q", len(vb)))
        fout.write(vb)
    elif kv_type in (GGUFValueType.UINT8, GGUFValueType.INT8, GGUFValueType.BOOL):
        fout.write(struct.pack("<B", value))
    elif kv_type in (GGUFValueType.UINT16, GGUFValueType.INT16):
        fout.write(struct.pack("<H", value))
    elif kv_type in (GGUFValueType.UINT32, GGUFValueType.INT32):
        fout.write(struct.pack("<I", value))
    elif kv_type == GGUFValueType.FLOAT32:
        fout.write(struct.pack("<f", value))
    elif kv_type in (GGUFValueType.UINT64, GGUFValueType.INT64):
        fout.write(struct.pack("<Q", value))
    elif kv_type == GGUFValueType.FLOAT64:
        fout.write(struct.pack("<d", value))


def write_kv_pair(fout, key, kv_type, value):
    kb = key.encode("utf-8")
    fout.write(struct.pack("<Q", len(kb)))
    fout.write(kb)
    fout.write(struct.pack("<I", int(kv_type)))
    if kv_type == GGUFValueType.STRING:
        write_kv_value(fout, kv_type, value)
    elif kv_type == GGUFValueType.ARRAY:
        sub_type, arr = value[0], value[1]
        fout.write(struct.pack("<I", int(sub_type)))
        fout.write(struct.pack("<Q", len(arr)))
        for elem in arr:
            write_kv_value(fout, sub_type, elem)
    else:
        write_kv_value(fout, kv_type, value)


def load_selection(path):
    """Return (per_layer dict[int,list] or None, global_keep list or None)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        per_layer = {}
        for k, v in data.items():
            try:
                layer = int(k)
            except (TypeError, ValueError):
                raise ValueError(f"Layer key must be an integer: {k!r}")
            if not isinstance(v, list):
                raise ValueError(f"Selection for layer {k} must be a list of expert ids")
            per_layer[layer] = [int(x) for x in v]
        return per_layer, None
    elif isinstance(data, list):
        gk = sorted(int(x) for x in data)
        if not gk:
            raise ValueError("Flat selection list is empty")
        return None, gk
    else:
        raise ValueError("Selection file must contain a JSON object or array")


def import_blk_layer(name):
    import re
    m = re.match(r"^blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def main() -> None:
    args = sys.argv[1:]
    verbose = False
    dry_run = False
    if "--dry-run" in args or "-n" in args:
        dry_run = True
        args = [a for a in args if a not in ("--dry-run", "-n")]
    if "--verbose" in args or "-v" in args:
        verbose = True
        args = [a for a in args if a not in ("--verbose", "-v")]

    if len(args) != 3:
        print(f"Usage: {sys.argv[0]} <selection.json> <source.imatrix.gguf> "
              f"<output.imatrix.gguf> [--verbose] [--dry-run]", file=sys.stderr)
        sys.exit(1)

    selection_path, source_path, output_path = args

    # ------------------------------------------------------------------
    # 1. Load expert selection
    # ------------------------------------------------------------------
    try:
        per_layer_keep, global_keep = load_selection(selection_path)
    except Exception as exc:
        print(f"ERROR: Could not read selection file '{selection_path}': {exc}",
              file=sys.stderr)
        sys.exit(1)

    if per_layer_keep is not None:
        counts = {len(v) for v in per_layer_keep.values()}
        print(f"Selection: {selection_path}")
        print(f"  Layers listed:   {len(per_layer_keep)}")
        if len(counts) > 1:
            print("ERROR: kept-expert count is not uniform across layers: "
                  f"{sorted(counts)}. The imatrix stores a single expert axis size; "
                  "all layers must keep the same number of experts.", file=sys.stderr)
            sys.exit(1)
        new_expert_count = next(iter(counts))
        print(f"  Kept experts/layer: {new_expert_count}")
    else:
        new_expert_count = len(global_keep)
        print(f"Selection: {selection_path}")
        print(f"  Global keep list applied to every block")
        print(f"  Kept experts/layer: {new_expert_count}")

    # ------------------------------------------------------------------
    # 2. Read source imatrix & infer expert count (no arch required)
    # ------------------------------------------------------------------
    print(f"\nReading source imatrix: {source_path}")
    reader = GGUFReader(source_path)

    # Infer the original expert count from `.counts` tensors: their trailing dim
    # is 1 for non-expert importance and n_experts for routed-expert importance.
    cand = []
    for t in reader.tensors:
        if t.name.endswith(".counts") and len(t.shape) >= 1:
            d = int(t.shape[-1])
            if d > 1:
                cand.append(d)
    # Fallback to in_sum2 trailing dims if no counts tensors available.
    if not cand:
        for t in reader.tensors:
            if t.name.endswith(".in_sum2") and len(t.shape) >= 2:
                cand.append(int(t.shape[-1]))

    if not cand:
        print("ERROR: Could not infer expert count from imatrix tensors "
              "(no `.counts`/`.in_sum2` tensors with an expert axis). "
              "This may not be a MoE imatrix.", file=sys.stderr)
        sys.exit(1)

    orig_expert_count = max(set(cand), key=cand.count)
    print(f"  Tensors:        {len(reader.tensors)}")
    print(f"  expert axis:    {orig_expert_count} -> {new_expert_count}")

    if new_expert_count > orig_expert_count:
        print("ERROR: selection keeps more experts than the imatrix has", file=sys.stderr)
        sys.exit(1)

    def keep_for_layer(layer):
        if per_layer_keep is not None:
            return per_layer_keep.get(layer)  # None -> layer untouched (keep all)
        return global_keep

    # Validate ids in range
    if per_layer_keep is not None:
        for layer, lst in per_layer_keep.items():
            bad = [e for e in lst if e < 0 or e >= orig_expert_count]
            if bad:
                print(f"ERROR: layer {layer} has expert ids out of range "
                      f"[0..{orig_expert_count - 1}]: {bad}", file=sys.stderr)
                sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Build output tensor plan (slice any tensor whose last dim == expert count)
    # ------------------------------------------------------------------
    plan = []
    total_keep_bytes = 0
    dropped_expert_bytes = 0
    sliced_tensor_count = 0
    touched_layers = set()

    for t in reader.tensors:
        shape = [int(d) for d in t.shape]
        layer = import_blk_layer(t.name)
        keep = keep_for_layer(layer) if layer is not None else None

        # Slice only genuine MoE routed-expert / router importance tensors whose
        # expert axis (last dim) equals the expert count. The name guard prevents
        # slicing unrelated arrays that merely share the size (e.g. attn_k/v with
        # key length == n_kv_heads*head_dim == expert_count in full-attn layers).
        lname = t.name.lower()
        looks_moe = (any(m in lname for m in MOE_NAME_MARKERS)
                     and not any(a in lname for a in ATTN_EXCLUDE))
        is_expert_axis = (len(shape) >= 1 and shape[-1] == orig_expert_count
                          and keep is not None and looks_moe)

        try:
            full_bytes = tensor_byte_size_from_shape(shape, t.tensor_type)
        except ValueError as exc:
            print(f"ERROR: {t.name}: {exc}", file=sys.stderr)
            sys.exit(1)

        if is_expert_axis:
            # bytes per expert block along the (slowest) last dim
            inner_elems = prod_int(shape[:-1])
            # ensure inner dims align to quant blocks so the expert block is whole
            bs, ts = GGML_QUANT_SIZES[int(t.tensor_type)]
            if inner_elems % bs != 0:
                print(f"ERROR: {t.name}: inner element count {inner_elems} not "
                      f"divisible by {bs}-element blocks; cannot byte-slice safely.",
                      file=sys.stderr)
                sys.exit(1)
            per_expert_bytes = (inner_elems // bs) * ts
            if full_bytes % orig_expert_count != 0 or per_expert_bytes * orig_expert_count != full_bytes:
                print(f"ERROR: {t.name}: byte size {full_bytes} inconsistent with "
                      f"expert count {orig_expert_count}", file=sys.stderr)
                sys.exit(1)
            new_shape = shape[:-1] + [len(keep)]
            new_bytes = per_expert_bytes * len(keep)
            plan.append({
                "t": t, "new_name": t.name, "new_shape": new_shape,
                "sliced": True, "keep": keep,
                "per_expert_bytes": per_expert_bytes,
                "out_bytes": new_bytes,
            })
            total_keep_bytes += new_bytes
            dropped_expert_bytes += (full_bytes - new_bytes)
            sliced_tensor_count += 1
            touched_layers.add(layer)
        else:
            plan.append({
                "t": t, "new_name": t.name, "new_shape": shape,
                "sliced": False, "keep": None, "out_bytes": full_bytes,
            })
            total_keep_bytes += full_bytes

    print(f"  Importance tensors sliced: {sliced_tensor_count}")
    print(f"  Layers touched:            {len(touched_layers)}")
    print(f"  Output tensor data size:   {fmt_size(total_keep_bytes)} "
          f"(dropped {fmt_size(dropped_expert_bytes)})")

    if verbose:
        for e in plan:
            if e["sliced"]:
                print(f"    slice {e['new_name']}: {list(e['t'].shape)} -> {e['new_shape']} "
                      f"(keep {len(e['keep'])}/{orig_expert_count})")

    # --- Audit: tensors whose last dim == expert_count but NOT sliced (the
    # 'coincidental 512' trap). Correctly skipped by the name guard, but list
    # them so you can confirm none are experts that should have been reaped.
    coincidental = [e for e in plan
                    if not e["sliced"] and len(e["new_shape"]) >= 1
                    and e["new_shape"][-1] == orig_expert_count]

    print("\n--- AUDIT: importance tensors with last dim == expert_count ---")
    import re as _re
    sliced_names = sorted(e["new_name"] for e in plan if e["sliced"])
    base_types = {}
    for n in sliced_names:
        key = _re.sub(r"^blk\.\d+\.", "", n)
        base_types[key] = base_types.get(key, 0) + 1
    print("  SLICED (treated as MoE expert/router importance), grouped by suffix:")
    for k in sorted(base_types):
        print(f"    {k}   ({base_types[k]} tensors)")
    if coincidental:
        print(f"  NOT sliced despite last dim == {orig_expert_count} "
              f"(name guard kept them — verify none are experts to reap):")
        cgroups = {}
        for e in coincidental:
            key = _re.sub(r"^blk\.\d+\.", "", e["new_name"])
            cgroups.setdefault(key, []).append(e["new_name"])
        for k in sorted(cgroups):
            names = cgroups[k]
            sample = names[0] + (f" ... ({len(names)} total)" if len(names) > 1 else "")
            print(f"    {sample}")
    else:
        print("  No other importance tensors share the expert-count size. Clean.")

    if dry_run:
        print("\n[dry-run] No file written.")
        return

    # ------------------------------------------------------------------
    # 4. Collect KV pairs verbatim (skip internal GGUF.* keys)
    # ------------------------------------------------------------------
    kv_pairs = []
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        kv_type = field.types[0]
        contents = field.contents()
        if kv_type == GGUFValueType.ARRAY:
            sub_type = field.types[1] if len(field.types) > 1 else GGUFValueType.FLOAT32
            kv_pairs.append((key, kv_type, (sub_type, contents)))
        else:
            kv_pairs.append((key, kv_type, contents))

    # ------------------------------------------------------------------
    # 5. Write output imatrix GGUF
    # ------------------------------------------------------------------
    print(f"\nWriting output: {output_path}")
    CHUNK = 64 * 1024 * 1024

    with open(source_path, "rb") as fin, open(output_path, "wb") as fout:
        fout.write(b"GGUF")
        fout.write(struct.pack("<I", 3))
        fout.write(struct.pack("<Q", len(plan)))
        fout.write(struct.pack("<Q", len(kv_pairs)))

        for key, kv_type, value in kv_pairs:
            write_kv_pair(fout, key, kv_type, value)

        # Determine alignment (llama.cpp requires every tensor offset to be a
        # multiple of general.alignment within the data section).
        alignment = 32
        try:
            f = reader.get_field("general.alignment")
            if f is not None:
                alignment = int(f.contents()) or 32
        except Exception:
            pass

        def align_up(n, a):
            return (n + a - 1) // a * a

        # Offsets are relative to the start of the (aligned) data section; each
        # tensor begins at an alignment boundary (llama.cpp validates this).
        cur = 0
        offsets = []
        for e in plan:
            offsets.append(cur)
            cur = align_up(cur + e["out_bytes"], alignment)

        for i, e in enumerate(plan):
            nb = e["new_name"].encode("utf-8")
            fout.write(struct.pack("<Q", len(nb)))
            fout.write(nb)
            fout.write(struct.pack("<I", len(e["new_shape"])))
            for dim in e["new_shape"]:
                fout.write(struct.pack("<Q", dim))
            fout.write(struct.pack("<I", int(e["t"].tensor_type)))
            fout.write(struct.pack("<Q", offsets[i]))

        # Pad header so the data section starts on an alignment boundary.
        pad = (alignment - (fout.tell() % alignment)) % alignment
        if pad:
            fout.write(b"\x00" * pad)
        data_start = fout.tell()

        # tensor data (slice kept experts' contiguous byte blocks).
        # After each tensor, pad so the next one begins on an alignment boundary,
        # matching the declared offsets.
        def write_zeros(n):
            while n > 0:
                chunk = min(CHUNK, n)
                fout.write(b"\x00" * chunk)
                n -= chunk

        print(f"Copying {len(plan)} tensors...")
        for i, e in enumerate(plan):
            t = e["t"]
            if e["sliced"]:
                peb = e["per_expert_bytes"]
                base = t.data_offset
                for exp_idx in e["keep"]:
                    fin.seek(base + exp_idx * peb)
                    remaining = peb
                    while remaining > 0:
                        buf = fin.read(min(CHUNK, remaining))
                        if not buf:
                            break
                        fout.write(buf)
                        remaining -= len(buf)
            else:
                fin.seek(t.data_offset)
                remaining = e["out_bytes"]
                while remaining > 0:
                    buf = fin.read(min(CHUNK, remaining))
                    if not buf:
                        break
                    fout.write(buf)
                    remaining -= len(buf)

            # pad to next aligned offset (relative to data_start)
            target = align_up((fout.tell() - data_start), alignment)
            gap = target - (fout.tell() - data_start)
            if gap:
                write_zeros(gap)

            if (i + 1) % 200 == 0 or i == len(plan) - 1:
                print(f"  Copied {i + 1}/{len(plan)} tensors")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    file_size = Path(source_path).stat().st_size
    output_size = Path(output_path).stat().st_size
    print(f"\nOutput: {output_path}")
    print(f"  Size:        {fmt_size(output_size)} "
          f"({100 * output_size / file_size:.1f}% of source, "
          f"saved {fmt_size(file_size - output_size)})")
    print(f"  Tensors:     {len(plan)} (sliced {sliced_tensor_count})")
    print(f"  expert axis: {new_expert_count} (was {orig_expert_count})")
    print(f"\nDone. Reaped imatrix importance arrays into: {output_path}")


if __name__ == "__main__":
    main()
