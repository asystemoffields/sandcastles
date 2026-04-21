"""
embedding.py -- Verilog generator for embedding (lookup table) layers.

For small vocabularies (``V * D`` below INLINE_THRESHOLD) the table is
emitted as an inline Verilog case statement inside a function — synthesis
tools infer ROM efficiently from that.

For large vocabularies (e.g. GPT-2's 50,257-entry embedding) we switch to
a ``$readmemh``-initialized reg array and write the hex file alongside the
Verilog.  A V-deep nested-ternary chain per output wire would produce tens
of millions of logic terms and crash yosys (or make synthesis take hours);
synthesis handles ``$readmemh`` ROM much better.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import math

from w2s.core import Operation, TensorWires
from w2s.emit import section_comment, slit


# If V*D (total table entries) is below this, emit inline case-statement ROM.
# Above it, use $readmemh with an external hex file.
INLINE_THRESHOLD = 1024


def _twos_complement_hex(val: int, bits: int) -> str:
    """Unsigned two's-complement hex rendering of a signed integer."""
    if val < 0:
        val = (1 << bits) + val
    n_hex = (bits + 3) // 4
    return f"{val:0{n_hex}X}"


def generate_embedding(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
    hex_dir: Optional[str] = None,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for an embedding lookup table.

    Weight shape : (num_embeddings, embedding_dim) — quantized integers.
    Input        : one or more unsigned index wires (seq_len positions).
    Output shape : (seq_len, embedding_dim).

    For small tables, emits an inline case-based ROM function.  For large
    tables, writes a hex file to ``hex_dir`` and emits a ``reg`` array with
    ``$readmemh``.  If ``hex_dir`` is None for a table that exceeds the
    inline threshold, falls back to the (much slower-to-synthesize) inline
    form with a warning comment.
    """
    weight = op.q_weights['weight']                     # (V, D) quantized
    V, D = weight.shape

    num_embeddings = op.attrs.get('num_embeddings', V)
    embedding_dim = op.attrs.get('embedding_dim', D)

    inp_name = op.inputs[0]
    inp_tw = wire_map[inp_name]
    seq_len = inp_tw.numel
    idx_bits = max(1, math.ceil(math.log2(max(num_embeddings, 2))))

    out_name = op.outputs[0]
    lines: List[str] = []

    total_entries = num_embeddings * embedding_dim
    use_hex_rom = total_entries > INLINE_THRESHOLD and hex_dir is not None

    lines += section_comment(
        f"{op.name}: Embedding  vocab={num_embeddings}  dim={embedding_dim}  "
        f"seq_len={seq_len}  ({'$readmemh ROM' if use_hex_rom else 'inline ROM'})"
    )
    lines.append(
        f"    // {num_embeddings} × {embedding_dim} = {total_entries:,} values hardwired"
    )
    if total_entries > INLINE_THRESHOLD and not use_hex_rom:
        lines.append(
            f"    // WARNING: table exceeds inline threshold ({INLINE_THRESHOLD}) "
            f"but no hex_dir was provided; emitting inline anyway — synthesis "
            f"may be extremely slow.  Pass output_dir through compile_graph."
        )
    lines.append("")

    out_wire_names: List[str] = []

    if use_hex_rom:
        # ---- $readmemh-initialized ROM -----------------------------------
        # Flat 1-D array of V*D entries in row-major order; address =
        # idx * D + d.  Writing the hex file here keeps it next to the .v.
        hex_path = Path(hex_dir) / f"{op.name}.hex"
        Path(hex_dir).mkdir(parents=True, exist_ok=True)
        with open(hex_path, "w", encoding="utf-8") as f:
            for row in weight:
                for val in row:
                    f.write(_twos_complement_hex(int(val), bits) + "\n")

        addr_bits = max(1, math.ceil(math.log2(max(total_entries, 2))))

        lines.append(
            f"    // ROM: {total_entries} entries, {bits} bits, "
            f"loaded from {hex_path.name} at elaboration"
        )
        lines.append(
            f"    reg signed [{bits - 1}:0] {op.name}_table [0:{total_entries - 1}];"
        )
        lines.append(f"    initial $readmemh(\"{hex_path.name}\", {op.name}_table);")
        lines.append("")

        # One wire per (pos, d) output, each a single ROM read.
        for pos in range(seq_len):
            idx_wire = inp_tw.wire_names[pos]
            lines.append(f"    // --- position {pos} ---")
            for d in range(embedding_dim):
                out_wire = f"{op.name}_out_{pos * embedding_dim + d}"
                out_wire_names.append(out_wire)
                # Address = idx * D + d, zero-extended to the ROM address width.
                if embedding_dim == 1:
                    addr_expr = f"{idx_wire}"
                else:
                    addr_expr = (
                        f"({{{{{addr_bits - idx_bits}'b0, {idx_wire}}}}} * "
                        f"{addr_bits}'d{embedding_dim}) + {addr_bits}'d{d}"
                    )
                lines.append(
                    f"    wire signed [{bits - 1}:0] {out_wire} = "
                    f"{op.name}_table[{addr_expr}];"
                )
            lines.append("")
    else:
        # ---- Inline case-in-function ROM ---------------------------------
        # Wrap the table in a Verilog function so synthesis infers one ROM
        # and reuses it across every lookup, instead of duplicating a
        # V-deep ternary chain per (pos, d).
        lines.append(
            f"    function signed [{bits - 1}:0] {op.name}_lookup;"
        )
        lines.append(f"        input [{idx_bits - 1}:0] row;")
        lines.append(f"        input [{max(1, math.ceil(math.log2(max(embedding_dim, 2)))) - 1}:0] col;")
        lines.append(f"        reg [{idx_bits + math.ceil(math.log2(max(embedding_dim, 2))) - 1}:0] addr;")
        lines.append(f"        begin")
        lines.append(f"            addr = row * {embedding_dim} + col;")
        lines.append(f"            case (addr)")
        for v in range(num_embeddings):
            for d in range(embedding_dim):
                addr = v * embedding_dim + d
                val = int(weight[v, d])
                lines.append(
                    f"                {addr}: {op.name}_lookup = {slit(bits, val)};"
                )
        lines.append(f"                default: {op.name}_lookup = {slit(bits, 0)};")
        lines.append(f"            endcase")
        lines.append(f"        end")
        lines.append(f"    endfunction")
        lines.append("")

        for pos in range(seq_len):
            idx_wire = inp_tw.wire_names[pos]
            lines.append(f"    // --- position {pos} ---")
            for d in range(embedding_dim):
                out_wire = f"{op.name}_out_{pos * embedding_dim + d}"
                out_wire_names.append(out_wire)
                lines.append(
                    f"    wire signed [{bits - 1}:0] {out_wire} = "
                    f"{op.name}_lookup({idx_wire}, {d});"
                )
            lines.append("")

    out_tw = TensorWires(
        wire_names=out_wire_names,
        shape=(seq_len, embedding_dim),
        bits=bits,
        signed=True,
    )

    return lines, {out_name: out_tw}
