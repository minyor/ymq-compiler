#!/usr/bin/env python3
"""
Reap (remove) MoE experts from a GGUF model, keeping ONLY the experts listed
per layer in a selection JSON file.

This is an EXPERT-level pruner (not a layer pruner). It slices the routed-expert
tensors along the expert axis and removes every expert that is not selected.

How it works (no requantization):
  - GGUF stores tensors column-major, so the LAST element dimension is the
    slowest-varying = a contiguous block of bytes in memory. In these models the
    routed-expert axis IS that last dimension (e.g. ffn_gate_exps.weight has
    shape [hidden, ffn_dim, n_experts]; the router ffn_gate_inp.weight has
    shape [hidden, n_experts]).
  - Therefore each expert is a contiguous run of bytes inside its tensor. We can
    keep only the selected experts by gathering their byte-blocks and
    concatenating them — exact same dtype / quant type, no precision loss.
  - The router (ffn_gate_inp.weight) is sliced with the SAME per-layer keep list,
    so surviving router columns stay index-aligned with the surviving expert
    arrays. No remap table is required.
  - Shared-expert tensors (e.g. *_shexp.*) have an expert axis != n_experts and
    are left untouched automatically.

Everything model-specific (architecture name, block_count, expert_count, tensor
shapes, dtypes) is read from the source GGUF at runtime — nothing is hardcoded,
so the script works across different MoE models.

Selection JSON format:
  A mapping of layer index -> list of EXPERT IDs to keep in that layer:
      {"0": [0, 1, 2, 4, ...], "1": [...], ..., "47": [...]}
  Or a single flat list applied identically to every block that has experts:
      [0, 1, 2, 4, ...]

  Layer keys / values are integers. The number of kept experts must be the same
  for every layer (GGUF represents expert_count as a single scalar).

Usage:
    python reap_gguf.py <selection.json> <source.gguf> <output.gguf> [--verbose]

Arguments:
    selection.json   JSON listing which experts to keep (per layer, or global)
    source.gguf      input GGUF model
    output.gguf      pruned GGUF with non-selected experts removed

Options:
    -v, --verbose    show per-tensor reaping details

Example:
    python reap_gguf.py seleccion_mass_K320.json model.gguf model-reaped.gguf
"""

import sys
import json
import struct
from pathlib import Path

from gguf import GGUFReader, GGUFValueType, GGML_QUANT_SIZES


# Substrings that identify MoE routed-expert / router importance tensors.
# Only tensors whose name contains one of these AND whose last dim equals the
# expert count are sliced. This prevents accidentally slicing unrelated tensors
# that merely share the same numeric size (e.g. attn_k/attn_v with key length ==
# n_kv_heads*head_dim == expert_count in full-attention layers).
MOE_NAME_MARKERS = ("_exps", "ffn_gate_inp", "expert")

# Substrings that mark attention tensors — never slice these even if a marker
# matched (defensive; attention weights are not MoE).
ATTN_EXCLUDE = ("attn_", ".qkv", "_q.", "_k.", "_v.", "_o.")


QUANT_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q4_K_S", 11: "Q4_K_M", 12: "Q5_K_S", 13: "Q5_K_M",
    14: "Q6_K", 17: "IQ2_XXS", 18: "IQ2_XS", 19: "IQ3_XS", 20: "IQ3_XXS",
    21: "IQ4_XS", 23: "IQ1_S", 24: "IQ4_NL", 25: "IQ2_S", 26: "IQ2_M", 27: "IQ3_S",
    29: "TQ1_0", 30: "TQ2_0",
}

PLAIN_DTYPES = {0, 1, 2, 5}  # F32, F16, BF16(no), F64 — element size == dtype itemsize


def get_field_value(reader, key):
    field = reader.get_field(key)
    return field.contents() if field else None


def quant_name(ttype):
    return QUANT_TYPE_NAMES.get(int(ttype), f"UNKNOWN({int(ttype)})")


def fmt_size(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def calculate_on_disk_sizes(tensors, file_size):
    """On-disk byte size of each tensor (incl. any per-row metadata / padding)."""
    n = len(tensors)
    sizes = []
    for i in range(n):
        if i < n - 1:
            sizes.append(tensors[i + 1].data_offset - tensors[i].data_offset)
        else:
            sizes.append(file_size - tensors[i].data_offset)
    return sizes


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
    """
    Load expert-selection.

    Returns:
      per_layer: dict[int, list[int]]  (layer -> kept expert ids), or
      global_keep: list[int]           when a flat list is given (applied to all blocks)
    """
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
            experts = [int(x) for x in v]
            per_layer[layer] = experts
        return per_layer, None
    elif isinstance(data, list):
        global_keep = sorted(int(x) for x in data)
        if not global_keep:
            raise ValueError("Flat selection list is empty")
        return None, global_keep
    else:
        raise ValueError("Selection file must contain a JSON object or array")


def element_count(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


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
        print(f"Usage: {sys.argv[0]} <selection.json> <source.gguf> <output.gguf> "
              f"[--verbose] [--dry-run]", file=sys.stderr)
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
            print(f"ERROR: kept-expert count is not uniform across layers: "
                  f"{sorted(counts)}. GGUF stores a single expert_count scalar; "
                  f"all layers must keep the same number of experts.", file=sys.stderr)
            sys.exit(1)
        new_expert_count = next(iter(counts))
        print(f"  Kept experts/layer: {new_expert_count}")
    else:
        new_expert_count = len(global_keep)
        print(f"Selection: {selection_path}")
        print(f"  Global keep list applied to every block")
        print(f"  Kept experts/layer: {new_expert_count}")

    # ------------------------------------------------------------------
    # 2. Read source GGUF & discover layout (nothing hardcoded)
    # ------------------------------------------------------------------
    print(f"\nReading source: {source_path}")
    reader = GGUFReader(source_path)
    file_size = Path(source_path).stat().st_size

    arch = get_field_value(reader, "general.architecture")
    if arch is None:
        print("ERROR: Source GGUF has no general.architecture key", file=sys.stderr)
        sys.exit(1)

    block_count = get_field_value(reader, f"{arch}.block_count")
    orig_expert_count = get_field_value(reader, f"{arch}.expert_count")

    # Fallback inference of expert count from *_exps tensor shapes if KV missing.
    if orig_expert_count is None:
        candidates = []
        for t in reader.tensors:
            if "_exps" in t.name and len(t.shape) >= 1:
                candidates.append(int(t.shape[-1]))
        if candidates:
            # most common trailing dim among exps tensors
            orig_expert_count = max(set(candidates), key=candidates.count)
            print(f"  (inferred expert_count from tensor shapes: {orig_expert_count})")
        else:
            print("ERROR: Cannot determine expert_count (no KV and no *_exps tensors). "
                  "This model may not be an MoE model.", file=sys.stderr)
            sys.exit(1)

    orig_expert_count = int(orig_expert_count)
    print(f"  Arch:           {arch}")
    print(f"  block_count:    {block_count}")
    print(f"  expert_count:   {orig_expert_count} -> {new_expert_count}")

    if new_expert_count > orig_expert_count:
        print("ERROR: selection keeps more experts than the model has", file=sys.stderr)
        sys.exit(1)

    # Validate kept ids in range
    def keep_for_layer(layer):
        if per_layer_keep is not None:
            lst = per_layer_keep.get(layer)
            if lst is None:
                return None  # layer not listed -> untouched (keep all experts)
            return lst
        return global_keep

    if per_layer_keep is not None:
        for layer, lst in per_layer_keep.items():
            bad = [e for e in lst if e < 0 or e >= orig_expert_count]
            if bad:
                print(f"ERROR: layer {layer} has expert ids out of range "
                      f"[0..{orig_expert_count - 1}]: {bad}", file=sys.stderr)
                sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Build output tensor plan
    # ------------------------------------------------------------------
    # Each entry: dict with source tensor, new_name (==old), new_shape, mode, keep list,
    #             per_expert_bytes (for sliced tensors), src_size
    on_disk_sizes = calculate_on_disk_sizes(reader.tensors, file_size)

    import re as _re
    blk_re = _re.compile(r"^blk\.(\d+)\.(.*)$")

    plan = []          # list of dicts describing each output tensor
    total_keep_bytes = 0
    dropped_expert_bytes = 0
    sliced_tensor_count = 0
    touched_layers = set()

    for t, src_size in zip(reader.tensors, on_disk_sizes):
        m = blk_re.match(t.name)
        layer = int(m.group(1)) if m else None
        shape = [int(d) for d in t.shape]

        keep = keep_for_layer(layer) if layer is not None else None
        # A tensor is an expert-axis tensor only if BOTH:
        #   - its LAST element dim == orig_expert_count (expert axis is slowest dim), AND
        #   - it actually looks like a MoE routed-expert / router tensor.
        # The name check is essential: non-MoE tensors can share the number 512
        # (e.g. attn_k/attn_v key length == n_kv_heads*head_dim == expert_count in
        # full-attention layers), and slicing those corrupts the model.
        lname = t.name.lower()
        looks_moe = (any(m in lname for m in MOE_NAME_MARKERS)
                     and not any(a in lname for a in ATTN_EXCLUDE))
        is_expert_axis = (len(shape) >= 1 and shape[-1] == orig_expert_count
                          and keep is not None and looks_moe)

        if is_expert_axis:
            new_shape = shape[:-1] + [len(keep)]
            # per-expert byte block = total tensor bytes / orig_expert_count
            # (robust for any dtype/quant, since expert axis is the outer/last dim)
            if src_size % orig_expert_count != 0:
                print(f"ERROR: tensor {t.name} size {src_size} not divisible by "
                      f"expert_count {orig_expert_count}; cannot byte-slice safely.",
                      file=sys.stderr)
                sys.exit(1)
            per_expert_bytes = src_size // orig_expert_count
            new_bytes = per_expert_bytes * len(keep)
            plan.append({
                "t": t, "new_name": t.name, "new_shape": new_shape,
                "sliced": True, "keep": keep,
                "per_expert_bytes": per_expert_bytes,
                "out_bytes": new_bytes,
            })
            total_keep_bytes += new_bytes
            dropped_expert_bytes += (src_size - new_bytes)
            sliced_tensor_count += 1
            touched_layers.add(layer)
        else:
            plan.append({
                "t": t, "new_name": t.name, "new_shape": shape,
                "sliced": False, "keep": None, "out_bytes": src_size,
            })
            total_keep_bytes += src_size

    print(f"  Expert-axis tensors sliced: {sliced_tensor_count}")
    print(f"  Layers touched:             {len(touched_layers)}")
    print(f"  Output tensor data size:    {fmt_size(total_keep_bytes)} "
          f"(dropped {fmt_size(dropped_expert_bytes)})")

    if verbose:
        for e in plan:
            if e["sliced"]:
                old = list(e["t"].shape)
                print(f"    slice {e['new_name']}: {old} -> {e['new_shape']} "
                      f"(keep {len(e['keep'])}/{orig_expert_count})")

    # --- Audit: surface tensors whose last dim == expert_count but were NOT
    # sliced (the 'coincidental 512' trap, e.g. attn_k key length). These are
    # correctly left untouched by the name guard, but listing them lets you
    # eyeball whether any of them is actually an expert tensor with unusual
    # naming that SHOULD have been sliced (-> add a MOE_NAME_MARKERS entry).
    coincidental = [e for e in plan
                    if not e["sliced"] and len(e["new_shape"]) >= 1
                    and e["new_shape"][-1] == orig_expert_count]

    print("\n--- AUDIT: tensors with last dim == expert_count ---")
    sliced_names = sorted(e["new_name"] for e in plan if e["sliced"])
    base_types = {}
    for n in sliced_names:
        import re as _re
        key = _re.sub(r"^blk\.\d+\.", "", n)
        key = _re.sub(r"^blk\.\d+$", "blk.N", key)
        base_types[key] = base_types.get(key, 0) + 1
    print("  SLICED (treated as MoE expert/router), grouped by suffix:")
    for k in sorted(base_types):
        print(f"    {k}   ({base_types[k]} tensors)")
    if coincidental:
        print(f"  NOT sliced despite last dim == {orig_expert_count} "
              f"(name guard kept them — verify none are experts to reap):")
        # group by suffix too, to keep output compact
        cgroups = {}
        for e in coincidental:
            import re as _re
            key = _re.sub(r"^blk\.\d+\.", "", e["new_name"])
            cgroups.setdefault(key, []).append(e["new_name"])
        for k in sorted(cgroups):
            names = cgroups[k]
            sample = names[0] + (f" ... ({len(names)} total)" if len(names) > 1 else "")
            print(f"    {sample}")
    else:
        print("  No other tensors share the expert-count size. Clean.")

    # Distinct suffixes sliced, for a quick sanity read of what got reaped.
    if dry_run:
        print("\n[dry-run] No file written.")
        return

    # ------------------------------------------------------------------
    # 4. Collect KV pairs (override expert_count if uniform)
    # ------------------------------------------------------------------
    kv_pairs = []
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key == f"{arch}.expert_count":
            kv_pairs.append((key, GGUFValueType.UINT32, new_expert_count))
            continue
        kv_type = field.types[0]
        contents = field.contents()
        if kv_type == GGUFValueType.ARRAY:
            sub_type = field.types[1] if len(field.types) > 1 else GGUFValueType.FLOAT32
            kv_pairs.append((key, kv_type, (sub_type, contents)))
        else:
            kv_pairs.append((key, kv_type, contents))

    # If the source had no expert_count KV but we sliced, add one.
    if not any(k == f"{arch}.expert_count" for k, _, _ in kv_pairs) and sliced_tensor_count:
        kv_pairs.append((f"{arch}.expert_count", GGUFValueType.UINT32, new_expert_count))

    # ------------------------------------------------------------------
    # 5. Write output GGUF
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

        alignment = get_field_value(reader, "general.alignment") or 32

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

        # tensor data
        def write_zeros(n):
            while n > 0:
                chunk = min(CHUNK, n)
                fout.write(b"\x00" * chunk)
                n -= chunk

        print(f"Copying {len(plan)} tensors...")
        for i, e in enumerate(plan):
            t = e["t"]
            if e["sliced"]:
                # gather only the kept experts' contiguous byte blocks
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

            if (i + 1) % 50 == 0 or i == len(plan) - 1:
                print(f"  Copied {i + 1}/{len(plan)} tensors")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    output_size = Path(output_path).stat().st_size
    print(f"\nOutput: {output_path}")
    print(f"  Size:        {fmt_size(output_size)} "
          f"({100 * output_size / file_size:.1f}% of source, "
          f"saved {fmt_size(file_size - output_size)})")
    print(f"  Tensors:     {len(plan)} (sliced {sliced_tensor_count})")
    print(f"  expert_count: {new_expert_count} (was {orig_expert_count})")
    print(f"\nDone. Reaped non-selected experts into: {output_path}")


if __name__ == "__main__":
    main()
