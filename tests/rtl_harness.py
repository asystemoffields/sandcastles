"""
rtl_harness.py — drive the *emitted* Verilog through Icarus Verilog and read
back the real hardware outputs, so tests can compare actual RTL behaviour
against the integer golden reference (forward_int) and/or a float reference.

This closes the gap that let silent-wrongness bugs hide: the test suite used to
check forward_int alone, never the Verilog that forward_int is supposed to
match.  Here we compile the graph to combinational Verilog, build a tiny
testbench, run vvp, and parse the output wires.

If iverilog/vvp are not installed the relevant tests skip (see have_iverilog()).
"""

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from w2s.graph import compile_graph, forward_int


def have_iverilog() -> bool:
    return bool(shutil.which("iverilog") and shutil.which("vvp"))


_PORT_RE = re.compile(
    r"(input|output)\s+wire\s+signed\s+\[(\d+):0\]\s+(\w+)"
)
_MODULE_RE = re.compile(r"module\s+(\w+)\s*\(")


def _parse_ports(vtext: str):
    """Return (module_name, [(name,width)] inputs, [(name,width)] outputs)."""
    mod = _MODULE_RE.search(vtext)
    if not mod:
        raise RuntimeError("could not find module declaration in emitted Verilog")
    name = mod.group(1)
    # Only scan the port list (up to the first ');').
    header = vtext.split(");", 1)[0]
    ins, outs = [], []
    for m in _PORT_RE.finditer(header):
        kind, hi, wn = m.group(1), int(m.group(2)), m.group(3)
        (ins if kind == "input" else outs).append((wn, hi + 1))
    return name, ins, outs


def quantize_inputs(graph, inputs):
    """Quantize float inputs to the integer domain exactly as forward_int does."""
    bits = graph.quant_config.bits
    qmax = 2 ** (bits - 1) - 1
    q = {}
    for name, data in inputs.items():
        scale = graph.tensor_scales.get(name, 1.0)
        q[name] = np.clip(np.round(np.asarray(data, dtype=np.float64) * scale),
                          -qmax, qmax).astype(np.int64)
    return q


def simulate(graph, float_inputs, workdir=None):
    """
    Compile *graph* to combinational Verilog, simulate it with the given inputs,
    and return {output_name: np.ndarray(int)} flattened per output tensor.

    float_inputs: dict name -> float array (same convention forward_int takes).
    """
    if not have_iverilog():
        raise RuntimeError("iverilog/vvp not available")

    cleanup = False
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="w2s_rtl_")
        cleanup = True
    try:
        vpath = compile_graph(graph, workdir, mode="combinational")
        vtext = open(vpath, encoding="utf-8").read()
        mod_name, ins, outs = _parse_ports(vtext)

        q_inputs = quantize_inputs(graph, float_inputs)

        # Flatten input values into wire-name -> value, matching the
        # `{name}_{i}` row-major flattening compile_graph uses.
        wire_vals = {}
        for name, arr in q_inputs.items():
            flat = np.asarray(arr).ravel()
            for i, v in enumerate(flat):
                wire_vals[f"{name}_{i}"] = int(v)

        # Build the testbench.
        tb = ["`timescale 1ns/1ps", "module tb;"]
        for wn, w in ins:
            tb.append(f"  reg signed [{w - 1}:0] {wn};")
        for wn, w in outs:
            tb.append(f"  wire signed [{w - 1}:0] {wn};")
        conns = ", ".join(f".{wn}({wn})" for wn, _ in ins + outs)
        tb.append(f"  {mod_name} dut({conns});")
        tb.append("  initial begin")
        for wn, _ in ins:
            val = wire_vals.get(wn, 0)
            tb.append(f"    {wn} = {val};")
        tb.append("    #10;")
        for wn, _ in outs:
            tb.append(f'    $display("{wn} %0d", $signed({wn}));')
        tb.append("    $finish;")
        tb.append("  end")
        tb.append("endmodule")
        tbpath = os.path.join(workdir, "tb.v")
        open(tbpath, "w").write("\n".join(tb))

        simbin = os.path.join(workdir, "sim.out")
        r = subprocess.run(
            ["iverilog", "-g2012", "-o", simbin, tbpath, vpath],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"iverilog failed:\n{r.stderr}\n{r.stdout}")
        r = subprocess.run(["vvp", simbin], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"vvp failed:\n{r.stderr}\n{r.stdout}")

        # Parse "wirename value" lines.
        vals = {}
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    vals[parts[0]] = int(parts[1])
                except ValueError:
                    pass

        # Group output wires by tensor name, preserving index order.
        result = {}
        for out_name in graph.output_names:
            multi = len(graph.output_names) > 1
            prefix = f"out_{out_name}_" if multi else "out_"
            idxs = []
            for wn, _ in outs:
                m = re.fullmatch(re.escape(prefix) + r"(\d+)", wn)
                if m:
                    idxs.append((int(m.group(1)), wn))
            idxs.sort()
            if idxs:
                result[out_name] = np.array([vals[wn] for _, wn in idxs], dtype=np.int64)
        return result
    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)


def assert_rtl_matches_golden(graph, float_inputs, atol=0, verbose=False):
    """
    Compare emitted-RTL outputs against forward_int (the integer golden ref).
    Returns (rtl, golden) dicts of flattened int arrays.  Raises AssertionError
    on mismatch beyond atol.
    """
    rtl = simulate(graph, float_inputs)
    golden = forward_int(graph, float_inputs)
    for name, gold in golden.items():
        g = np.asarray(gold).ravel()
        r = np.asarray(rtl[name]).ravel()
        if verbose:
            print(f"{name}: rtl={r.tolist()} golden={g.tolist()}")
        assert r.shape == g.shape, f"{name}: shape {r.shape} != golden {g.shape}"
        diff = np.abs(r - g)
        assert np.all(diff <= atol), (
            f"{name}: RTL != golden (max |diff|={diff.max()}, atol={atol})\n"
            f"  rtl   ={r.tolist()}\n  golden={g.tolist()}"
        )
    return rtl, golden
