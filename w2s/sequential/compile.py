"""
compile.py — Sequential architecture compiler.

Generates a clocked Verilog design where:
  - Weights are hardwired ROM (case-statement lookup — still the silicon)
  - A single MAC unit processes one weight×input per clock cycle
  - Register-file buffers hold intermediate activations
  - A state machine sequences through layers

This trades throughput for area. A 784→128→10 network uses ONE multiplier
instead of 100K parallel multipliers. It takes more cycles but fits on a
Tiny Tapeout tile.

The weights are still the silicon — they're constants in the ROM.
The compute is sequential. The area is minimal.
"""

import copy
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from w2s.core import ComputeGraph, Operation, OpType
from w2s import emit


# ---------------------------------------------------------------------------
#  Fusion pass: merge Dense + ReLU into Dense(activation='relu')
# ---------------------------------------------------------------------------

def _fuse_ops(ops: List[Operation]) -> List[Operation]:
    """Fuse activation ops into preceding weighted ops where possible."""
    # Deep-copy so we don't mutate the caller's original Operation objects
    ops = [copy.deepcopy(op) for op in ops]
    fused = []
    skip_next = False
    for i, op in enumerate(ops):
        if skip_next:
            skip_next = False
            continue
        if (op.op_type in (OpType.DENSE, OpType.CONV2D)
                and i + 1 < len(ops)
                and ops[i + 1].op_type == OpType.RELU
                and ops[i + 1].inputs[0] == op.outputs[0]):
            op.attrs['activation'] = 'relu'
            op.outputs = ops[i + 1].outputs  # rewire
            skip_next = True
        fused.append(op)
    return fused


# ---------------------------------------------------------------------------
#  Identify which ops are "compute layers" (have weights, need MAC)
# ---------------------------------------------------------------------------

# Ops the sequential state machine actually iterates correctly.  Historically
# this set also contained CONV1D/CONV2D, but the state machine only advances
# ``in_idx`` up to C_in per output neuron — it doesn't loop over kernel
# positions or spatial output pixels, so Conv sequential output is wrong by
# a factor of kH·kW·H_out·W_out.  Rather than silently emit wrong Verilog we
# reject Conv in sequential mode; users should compile CNNs in combinational.
_WEIGHTED_OPS = {OpType.DENSE}

# Structural ops that carry no runtime cost for the sequential pipeline
# (they're pure data-layout rewrites that happen implicitly because every
# buffer is already flat).  Anything NOT in this set and NOT weighted is a
# real op that sequential mode does not implement — we refuse to silently
# drop it.
_SEQUENTIAL_PASSTHROUGH = {OpType.RESHAPE, OpType.FLATTEN}


def _compute_layers(ops: List[Operation]) -> List[Operation]:
    """
    Return only ops that need MAC computation.

    Raises if the op stream contains any op that is neither a weighted
    compute layer nor a known structural passthrough — sequential mode can
    only faithfully compile MLP-shaped graphs (Dense with fused ReLU) and
    silently dropping MaxPool/LayerNorm/Conv/Softmax/etc. would produce
    Verilog that implements a different network than the graph.
    """
    unsupported: List[Tuple[str, str]] = []
    for op in ops:
        if op.op_type in _WEIGHTED_OPS:
            continue
        if op.op_type in _SEQUENTIAL_PASSTHROUGH:
            continue
        unsupported.append((op.name, op.op_type.value))
    if unsupported:
        items = ", ".join(f"{n}({t})" for n, t in unsupported)
        raise NotImplementedError(
            f"Sequential compile does not support these ops: {items}.  "
            f"Sequential mode currently handles Dense (with optionally "
            f"fused ReLU), plus Reshape/Flatten as layout no-ops.  For "
            f"Conv1D/Conv2D or any other op, use --mode combinational."
        )
    return [op for op in ops if op.op_type in _WEIGHTED_OPS]


# ---------------------------------------------------------------------------
#  Weight / bias ROM generation
# ---------------------------------------------------------------------------

# If a ROM has more entries than this, use $readmemh (external hex file)
# instead of an inline case statement. Keeps Verilog files manageable.
INLINE_ROM_THRESHOLD = 4096


def _to_twos_complement_hex(val: int, bits: int) -> str:
    """Convert signed int to two's complement hex string."""
    if val < 0:
        val = (1 << bits) + val
    n_hex = (bits + 3) // 4
    return f"{val:0{n_hex}X}"


def _write_hex_file(path: Path, values: np.ndarray, bits: int):
    """Write a ROM hex file (one value per line, two's complement)."""
    with open(path, "w") as f:
        for v in values.flatten():
            f.write(_to_twos_complement_hex(int(v), bits) + "\n")


def _weight_rom_lines(
    name: str, weights: np.ndarray, bits: int,
    hex_dir: Optional[Path] = None,
) -> List[str]:
    """
    Generate a ROM for weights.

    Small ROMs: inline case statement (constants in the Verilog itself).
    Large ROMs: $readmemh from external hex file (synthesis reads the file
    and bakes the values into the ROM — still hardwired silicon).
    """
    flat = weights.flatten()
    n = len(flat)
    addr_bits = max(math.ceil(math.log2(max(n, 2))), 1)
    L = []

    if n <= INLINE_ROM_THRESHOLD or hex_dir is None:
        # Inline case statement
        L.append(f"    // Weight ROM: {name} ({n:,} entries, inline)")
        L.append(f"    function signed [{bits - 1}:0] {name}_w;")
        L.append(f"        input [{addr_bits - 1}:0] addr;")
        L.append(f"        case (addr)")
        for i, v in enumerate(flat):
            L.append(f"            {addr_bits}'d{i}: {name}_w = {emit.slit(bits, int(v))};")
        L.append(f"            default: {name}_w = {bits}'sd0;")
        L.append(f"        endcase")
        L.append(f"    endfunction")
    else:
        # External hex file + $readmemh
        hex_path = hex_dir / f"{name}_weights.hex"
        _write_hex_file(hex_path, flat, bits)
        L.append(f"    // Weight ROM: {name} ({n:,} entries, from {hex_path.name})")
        L.append(f"    reg signed [{bits - 1}:0] {name}_wmem [0:{n - 1}];")
        L.append(f'    initial $readmemh("{hex_path.name}", {name}_wmem);')
        L.append(f"    // synthesis tool reads hex file and bakes values into ROM")
        L.append(f"    // THE WEIGHTS ARE STILL THE SILICON")
        # Wrap in a function for consistent interface
        L.append(f"    function signed [{bits - 1}:0] {name}_w;")
        L.append(f"        input [{addr_bits - 1}:0] addr;")
        L.append(f"        {name}_w = {name}_wmem[addr];")
        L.append(f"    endfunction")

    L.append("")
    return L


def _bias_rom_lines(
    name: str, biases: np.ndarray, acc_bits: int = 32,
    hex_dir: Optional[Path] = None,
) -> List[str]:
    """Generate a ROM for biases (in accumulator scale)."""
    n = len(biases)
    addr_bits = max(math.ceil(math.log2(max(n, 2))), 1)
    L = []

    if n <= INLINE_ROM_THRESHOLD or hex_dir is None:
        L.append(f"    // Bias ROM: {name} ({n} entries, {acc_bits}-bit)")
        L.append(f"    function signed [{acc_bits - 1}:0] {name}_b;")
        L.append(f"        input [{addr_bits - 1}:0] addr;")
        L.append(f"        case (addr)")
        for i, v in enumerate(biases):
            L.append(f"            {addr_bits}'d{i}: {name}_b = {emit.slit(acc_bits, int(v))};")
        L.append(f"            default: {name}_b = {acc_bits}'sd0;")
        L.append(f"        endcase")
        L.append(f"    endfunction")
    else:
        hex_path = hex_dir / f"{name}_biases.hex"
        _write_hex_file(hex_path, biases, acc_bits)
        L.append(f"    // Bias ROM: {name} ({n} entries, from {hex_path.name})")
        L.append(f"    reg signed [{acc_bits - 1}:0] {name}_bmem [0:{n - 1}];")
        L.append(f'    initial $readmemh("{hex_path.name}", {name}_bmem);')
        L.append(f"    function signed [{acc_bits - 1}:0] {name}_b;")
        L.append(f"        input [{addr_bits - 1}:0] addr;")
        L.append(f"        {name}_b = {name}_bmem[addr];")
        L.append(f"    endfunction")

    L.append("")
    return L


# ---------------------------------------------------------------------------
#  Buffer declarations
# ---------------------------------------------------------------------------

def _buffer_decl(name: str, size: int, bits: int) -> List[str]:
    """Declare a register-file buffer."""
    return [f"    reg signed [{bits - 1}:0] {name} [0:{size - 1}];"]


# ---------------------------------------------------------------------------
#  Main sequential compiler
# ---------------------------------------------------------------------------

def compile_sequential(
    graph: ComputeGraph,
    output_dir: str = ".",
) -> str:
    """
    Compile a quantized ComputeGraph to a sequential Verilog design.

    Returns path to the generated .v file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bits = graph.quant_config.bits
    qmax = 2 ** (bits - 1) - 1

    # Fuse and sort ops
    ops = _fuse_ops(graph.topological_order())
    layers = _compute_layers(ops)

    if not layers:
        raise ValueError("No weighted layers found — nothing to compile sequentially")

    # Determine buffer sizes from layer shapes
    # buf[0] = graph input, buf[i+1] = output of layer i
    buf_sizes = []

    # Input buffer size
    n_input = 1
    for inp_name in graph.input_names:
        shape = graph.input_shapes.get(inp_name, (1,))
        s = 1
        for d in shape:
            s *= d
        n_input = s
    buf_sizes.append(n_input)

    # Layer output sizes (also buffer sizes)
    for layer in layers:
        if layer.op_type == OpType.DENSE:
            n_out = layer.q_weights['weight'].shape[0]
        elif layer.op_type == OpType.CONV2D:
            w = layer.q_weights['weight']
            c_out = w.shape[0]
            # Need to know spatial output dims — get from attrs or infer
            n_out = c_out  # simplified; full conv support needs spatial dims
        else:
            n_out = n_input  # fallback
        buf_sizes.append(n_out)

    n_output = buf_sizes[-1]
    n_layers = len(layers)

    # Compute address widths
    def aw(size):
        return max(math.ceil(math.log2(max(size, 2))), 1)

    # State enumeration
    state_bits = max(math.ceil(math.log2(3 + 2 * n_layers + 1)), 1)
    states = {}
    states['IDLE'] = 0
    states['LOAD'] = 1
    idx = 2
    for i in range(n_layers):
        states[f'L{i}_MAC'] = idx; idx += 1
        states[f'L{i}_WRITE'] = idx; idx += 1
    states['OUTPUT'] = idx; idx += 1
    states['DONE_ST'] = idx

    # Collect requant params.  Sequential mode uses a single MAC with a
    # single per-layer multiplier/shift, so it only supports PER-TENSOR
    # quantization.  Per-channel requant would need a per-output-neuron ROM
    # indexed by `out_neuron`; until that lands we refuse rather than
    # silently averaging the channel-specific constants (which is wrong,
    # especially for shifts which are exponents).
    requant_mults = []
    requant_shifts = []
    activations = []
    for layer in layers:
        rm = layer.q_params.get('requant_mult', 1)
        rs = layer.q_params.get('requant_shift', 16)
        if isinstance(rm, np.ndarray) and rm.size > 1:
            raise NotImplementedError(
                f"Sequential compile of layer {layer.name!r} has a "
                f"per-channel requant multiplier (shape={rm.shape}); "
                f"sequential mode currently requires per-tensor "
                f"quantization.  Re-run with QuantGranularity.PER_TENSOR, "
                f"or compile in --mode combinational."
            )
        if isinstance(rs, np.ndarray) and rs.size > 1:
            raise NotImplementedError(
                f"Sequential compile of layer {layer.name!r} has a "
                f"per-channel requant shift; see the per-channel note above."
            )
        if isinstance(rm, np.ndarray):
            rm = rm.item()
        if isinstance(rs, np.ndarray):
            rs = rs.item()
        requant_mults.append(int(rm))
        requant_shifts.append(int(rs))

        activations.append(layer.attrs.get('activation', 'none'))

    # Use common shift if possible
    common_shift = requant_shifts[0] if len(set(requant_shifts)) == 1 else None

    # ---------- Generate Verilog ----------
    L: List[str] = []
    def e(s=""):
        L.append(s)

    name = f"{graph.name}_seq"
    total_params = sum(
        int(np.prod(w.shape))
        for layer in layers
        for w in layer.q_weights.values()
    )

    # Header
    e(f"// {'=' * 75}")
    e(f"// {name} — Sequential Neural Network with Hardwired Weights")
    e(f"// Generated by weights2silicon (w2s)")
    e(f"//")
    e(f"// Architecture : SEQUENTIAL (one MAC per clock cycle)")
    e(f"// Weights      : {total_params:,} constants in ROM (THE SILICON)")
    e(f"// Layers       : {n_layers}")
    e(f"// Buffers      : {' -> '.join(str(s) for s in buf_sizes)}")
    e(f"// Quantization : int{bits}")
    e(f"//")
    e(f"// The weights are ROM constants — synthesis maps them into fixed logic.")
    e(f"// The compute is sequential — one multiply-accumulate per clock.")
    e(f"// Variable-length input: feed fewer/more values via serial I/O.")
    e(f"// {'=' * 75}")
    e()

    # Module declaration
    e(f"module {name} (")
    e(f"    input  wire       clk,")
    e(f"    input  wire       rst_n,")
    e(f"    input  wire signed [{bits - 1}:0] data_in,")
    e(f"    input  wire       data_valid,")
    e(f"    output reg  signed [{bits - 1}:0] data_out,")
    e(f"    output reg        out_valid,")
    e(f"    output reg        done,")
    e(f"    output wire       ready")
    e(f");")
    e()

    # ---- Activation buffers ----
    e(f"    // {'=' * 71}")
    e(f"    // Activation Buffers (register files)")
    e(f"    // {'=' * 71}")
    for i, sz in enumerate(buf_sizes):
        label = "input" if i == 0 else f"layer{i-1} output"
        e(f"    // buf{i}: {label} ({sz} elements)")
        e(f"    reg signed [{bits - 1}:0] buf{i} [0:{sz - 1}];")
    e()

    # ---- Weight and bias ROMs ----
    e(f"    // {'=' * 71}")
    e(f"    // Weight ROMs — THE WEIGHTS ARE THE SILICON")
    e(f"    // {'=' * 71}")
    for i, layer in enumerate(layers):
        w = layer.q_weights['weight']
        L.extend(_weight_rom_lines(f"l{i}", w, bits, hex_dir=out))
        b = layer.q_weights.get('bias')
        if b is not None:
            L.extend(_bias_rom_lines(f"l{i}", b, 32, hex_dir=out))
        else:
            # Zero bias ROM
            n_out = w.shape[0]
            zero_b = np.zeros(n_out, dtype=np.int64)
            L.extend(_bias_rom_lines(f"l{i}", zero_b, 32, hex_dir=out))

    # ---- MAC engine ----
    e(f"    // {'=' * 71}")
    e(f"    // MAC Engine + Requantization")
    e(f"    // {'=' * 71}")
    e(f"    reg signed [31:0] mac_acc;")
    e()

    # Requantization: mux the multiplier based on current layer
    max_idx_bits = max(aw(s) for s in buf_sizes) + 1
    e(f"    // Layer tracking")
    e(f"    reg [{state_bits - 1}:0] state;")
    e(f"    reg [{max_idx_bits - 1}:0] in_idx;")
    e(f"    reg [{max_idx_bits - 1}:0] out_neuron;")
    e(f"    reg [{max_idx_bits - 1}:0] io_count;")
    e()

    # Requantization wires
    e(f"    // Requantization (per-layer multiplier)")
    e(f"    reg signed [31:0] req_mult;")
    if common_shift is not None:
        shift_val = common_shift
    else:
        e(f"    reg [5:0] req_shift;")
        shift_val = None
    e(f"    wire signed [63:0] req_ext = {{{{32{{mac_acc[31]}}}}, mac_acc}};")
    e(f"    wire signed [63:0] req_prod = req_ext * {{{{32{{req_mult[31]}}}}, req_mult}};")
    if shift_val is not None:
        e(f"    wire signed [63:0] req_shifted = req_prod >>> {shift_val};")
    else:
        e(f"    wire signed [63:0] req_shifted = req_prod >>> req_shift;")
    e()
    e(f"    // Saturate (linear)")
    e(f"    wire signed [{bits - 1}:0] sat_linear =")
    e(f"        (req_shifted > 64'sd{qmax})  ? {emit.slit(bits, qmax)} :")
    e(f"        (req_shifted < -64'sd{qmax}) ? {emit.slit(bits, -qmax)} :")
    e(f"        req_shifted[{bits - 1}:0];")
    e()
    e(f"    // Saturate (ReLU)")
    e(f"    wire signed [{bits - 1}:0] sat_relu =")
    e(f"        (req_shifted > 64'sd{qmax}) ? {emit.slit(bits, qmax)} :")
    e(f"        (req_shifted < 64'sd0)      ? {bits}'sd0 :")
    e(f"        req_shifted[{bits - 1}:0];")
    e()

    # State parameters
    e(f"    // States")
    for sname, sval in states.items():
        e(f"    localparam {sname:12s} = {state_bits}'d{sval};")
    e()

    e(f"    assign ready = (state == IDLE);")
    e()

    # ---- Weight ROM read wires ----
    # For each layer, we need a wire driven by its weight ROM function
    for i, layer in enumerate(layers):
        w = layer.q_weights['weight']
        n_in_l = w.shape[1]
        n_out_l = w.shape[0]
        total_w = n_in_l * n_out_l
        waddr_bits = max(math.ceil(math.log2(max(total_w, 2))), 1)
        e(f"    // Layer {i} ROM readout")
        e(f"    reg [{waddr_bits - 1}:0] l{i}_waddr;")
        e(f"    wire signed [{bits - 1}:0] l{i}_wdata = l{i}_w(l{i}_waddr);")
        e(f"    wire signed [31:0] l{i}_bdata = l{i}_b(out_neuron[{aw(n_out_l) - 1}:0]);")
        e()

    # ---- Main state machine ----
    e(f"    // {'=' * 71}")
    e(f"    // Controller State Machine")
    e(f"    // {'=' * 71}")
    e(f"    always @(posedge clk or negedge rst_n) begin")
    e(f"        if (!rst_n) begin")
    e(f"            state     <= IDLE;")
    e(f"            done      <= 1'b0;")
    e(f"            out_valid <= 1'b0;")
    e(f"            mac_acc   <= 32'sd0;")
    e(f"            in_idx    <= 0;")
    e(f"            out_neuron <= 0;")
    e(f"            io_count  <= 0;")
    e(f"            req_mult  <= 32'sd1;")
    if shift_val is None:
        e(f"            req_shift <= 6'd16;")
    e(f"        end else begin")
    e(f"            case (state)")
    e()

    # IDLE
    e(f"                IDLE: begin")
    e(f"                    done      <= 1'b0;")
    e(f"                    out_valid <= 1'b0;")
    e(f"                    if (data_valid) begin")
    e(f"                        buf0[0]  <= data_in;")
    e(f"                        io_count <= {max_idx_bits}'d1;")
    e(f"                        state    <= LOAD;")
    e(f"                    end")
    e(f"                end")
    e()

    # LOAD
    in_max = n_input - 1
    e(f"                LOAD: begin")
    e(f"                    if (data_valid) begin")
    e(f"                        buf0[io_count[{aw(n_input) - 1}:0]] <= data_in;")
    e(f"                        if (io_count == {max_idx_bits}'d{in_max}) begin")
    # Transition to first layer MAC
    first_layer = layers[0]
    n_in_0 = first_layer.q_weights['weight'].shape[1]
    n_out_0 = first_layer.q_weights['weight'].shape[0]
    e(f"                            // Start layer 0")
    e(f"                            state      <= L0_MAC;")
    e(f"                            out_neuron <= 0;")
    e(f"                            in_idx     <= 0;")
    e(f"                            l0_waddr   <= 0;")
    e(f"                            mac_acc    <= l0_b(0);  // explicit index: out_neuron not yet 0")
    e(f"                            req_mult   <= 32'sd{requant_mults[0]};")
    if shift_val is None:
        e(f"                            req_shift  <= 6'd{requant_shifts[0]};")
    e(f"                        end else begin")
    e(f"                            io_count <= io_count + 1;")
    e(f"                        end")
    e(f"                    end")
    e(f"                end")
    e()

    # Per-layer MAC and WRITE states
    for li, layer in enumerate(layers):
        w = layer.q_weights['weight']
        n_in_l = w.shape[1]
        n_out_l = w.shape[0]
        in_buf = f"buf{li}"
        out_buf = f"buf{li + 1}"
        activation = activations[li]
        is_last = (li == n_layers - 1)

        # MAC state
        e(f"                L{li}_MAC: begin")
        e(f"                    // Layer {li}: MAC — acc += weight * input")
        in_addr_bits = aw(buf_sizes[li])
        e(f"                    mac_acc <= mac_acc + (l{li}_wdata * {in_buf}[in_idx[{in_addr_bits - 1}:0]]);")
        e(f"                    if (in_idx == {max_idx_bits}'d{n_in_l - 1}) begin")
        e(f"                        state <= L{li}_WRITE;")
        e(f"                    end else begin")
        e(f"                        in_idx   <= in_idx + 1;")
        e(f"                        l{li}_waddr <= l{li}_waddr + 1;")
        e(f"                    end")
        e(f"                end")
        e()

        # WRITE state
        out_addr_bits = aw(buf_sizes[li + 1])
        e(f"                L{li}_WRITE: begin")
        e(f"                    // Layer {li}: requantize + {'ReLU' if activation == 'relu' else 'linear'} → buffer")
        sat_wire = "sat_relu" if activation == 'relu' else "sat_linear"
        e(f"                    {out_buf}[out_neuron[{out_addr_bits - 1}:0]] <= {sat_wire};")

        if is_last:
            # Last layer — go to OUTPUT
            e(f"                    if (out_neuron == {max_idx_bits}'d{n_out_l - 1}) begin")
            e(f"                        state    <= OUTPUT;")
            e(f"                        io_count <= 0;")
            e(f"                    end else begin")
        else:
            # Not last — start next neuron or next layer
            next_layer = layers[li + 1]
            nw = next_layer.q_weights['weight']
            n_in_next = nw.shape[1]
            e(f"                    if (out_neuron == {max_idx_bits}'d{n_out_l - 1}) begin")
            e(f"                        // Start layer {li + 1}")
            e(f"                        state      <= L{li + 1}_MAC;")
            e(f"                        out_neuron <= 0;")
            e(f"                        in_idx     <= 0;")
            e(f"                        l{li + 1}_waddr <= 0;")
            e(f"                        mac_acc    <= l{li + 1}_b(0);  // explicit index: out_neuron not yet 0")
            e(f"                        req_mult   <= 32'sd{requant_mults[li + 1]};")
            if shift_val is None:
                e(f"                        req_shift  <= 6'd{requant_shifts[li + 1]};")
            e(f"                    end else begin")

        # Common: next neuron in same layer
        e(f"                        out_neuron <= out_neuron + 1;")
        e(f"                        in_idx     <= 0;")
        # Use the weight-address register width for the constant literal
        # so that n_in_l is not truncated when it exceeds 2^max_idx_bits.
        # Also zero-extend out_neuron to waddr_bits so the multiplication
        # is performed at sufficient width and does not overflow.
        total_w_l = n_in_l * n_out_l
        waddr_bits_l = max(math.ceil(math.log2(max(total_w_l, 2))), 1)
        pad = waddr_bits_l - max_idx_bits
        if pad > 0:
            neuron_ext = f"{{{{{pad}'b0, out_neuron}}}}"
        else:
            neuron_ext = "out_neuron"
        waddr_expr = f"({neuron_ext} + {waddr_bits_l}'d1) * {waddr_bits_l}'d{n_in_l}"
        e(f"                        l{li}_waddr <= {waddr_expr};")
        # Bias for next neuron: out_neuron + 1 is safe here because
        # this else-branch only runs when out_neuron < n_out_l - 1
        e(f"                        mac_acc    <= l{li}_b(out_neuron[{aw(n_out_l) - 1}:0] + 1);")
        e(f"                        state      <= L{li}_MAC;")
        e(f"                    end")
        e(f"                end")
        e()

    # OUTPUT
    out_buf_name = f"buf{n_layers}"
    out_addr_bits = aw(n_output)
    e(f"                OUTPUT: begin")
    e(f"                    data_out  <= {out_buf_name}[io_count[{out_addr_bits - 1}:0]];")
    e(f"                    out_valid <= 1'b1;")
    e(f"                    if (io_count == {max_idx_bits}'d{n_output - 1}) begin")
    e(f"                        state     <= DONE_ST;")
    e(f"                    end else begin")
    e(f"                        io_count  <= io_count + 1;")
    e(f"                    end")
    e(f"                end")
    e()

    # DONE
    e(f"                DONE_ST: begin")
    e(f"                    out_valid  <= 1'b0;")
    e(f"                    done       <= 1'b1;")
    e(f"                    out_neuron <= 0;  // reset for next inference")
    e(f"                    state      <= IDLE;")
    e(f"                end")
    e()
    e(f"            endcase")
    e(f"        end")
    e(f"    end")
    e()
    e(f"endmodule")

    # Write file
    vpath = out / f"{name}.v"
    vpath.write_text("\n".join(L), encoding="utf-8")

    # Summary
    total_cycles = sum(
        layer.q_weights['weight'].shape[0] * layer.q_weights['weight'].shape[1]
        for layer in layers
    )
    print(f"  Sequential mode: {total_params:,} weights in ROM, "
          f"~{total_cycles:,} cycles per inference")

    return str(vpath)
