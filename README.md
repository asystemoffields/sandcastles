# sandcastles

**Compile neural network weights directly into synthesizable Verilog.**

```
Trained model (.onnx, HuggingFace, numpy)
        |
  [ sandcastles ]
        |
  Synthesizable Verilog (the hardwired weights)
        |
   Yosys + nextpnr (FPGA) / OpenLane (ASIC)
        |
  Bitstream or GDS-II layout
        |
   iCE40 / ECP5 / Tiny Tapeout / fab
        |
  YOUR CHIP
```

## What this does

sandcastles takes a trained neural network, quantizes the weights to fixed-point integers, and generates Verilog where **every weight is a numeric constant in the logic**. When a synthesis tool processes this Verilog, each `constant * input` multiplication becomes a fixed shift-add circuit. The weight values literally determine the transistor topology.

This is the same core idea behind [Taalas](https://taalas.com/), which hard-wires Llama 3.1 8B into a physical chip achieving 17,000 tokens/sec.

## Supported operations

| Category | Operations |
|----------|-----------|
| **Linear** | Dense (fully connected) |
| **Convolution** | Conv1D, Conv2D |
| **Activation** | ReLU, GELU, Sigmoid, Tanh, SiLU, Softmax |
| **Normalization** | LayerNorm, RMSNorm, BatchNorm |
| **Attention** | Multi-Head Attention, Grouped-Query Attention (GQA) |
| **Transformer** | SwiGLU FFN, RoPE, KV Cache |
| **Embedding** | Token embedding (hardwired lookup table) |
| **Pooling** | MaxPool2D, AvgPool2D, GlobalAvgPool |
| **Structural** | Add (residual), Multiply, Reshape, Flatten, Concat |

Architecturally supports the building blocks used in modern LLMs (DeepSeek, Llama, Qwen, Mistral, Gemma). Tested end-to-end with GPT-2.

**Quantization**: int4, int8, int16. Symmetric or asymmetric. Per-tensor or per-channel. Mixed-precision per layer.

**Import**: HuggingFace models (GPT-2, Llama, Mistral, Qwen2, Phi, Gemma), ONNX models, numpy arrays, or the fluent GraphBuilder API.

**Output**: Combinational core, sequential ROM+MAC, serial I/O wrapper, Tiny Tapeout interface.

**Compilation modes**:
- *Combinational* — every weight gets its own multiplier. Max speed, max area.
- *Sequential* — one multiplier + weight ROM + state machine. Minimal area, fits Tiny Tapeout.

**Sparsity**: Zero weights are automatically eliminated from the netlist. Supports pruning to target sparsity, structured 2:4 sparsity enforcement, and sparsity-aware area estimation.

**FPGA targeting**: Resource estimation for iCE40 and ECP5 devices (LUTs, BRAM, DSP slices). Generates Makefiles for open-source toolchains (Yosys + nextpnr) and pin constraint templates.

**Testbench generation**: Automated Verilog testbenches with golden vectors from the quantized model, VCD waveform dump, tolerance checking, and sequential mode support.

**Auto-fit**: Given a model and a target FPGA device, automatically finds the best quantization, mixed precision, sparsity, and compilation mode to make the design fit. Runs per-layer sensitivity analysis, then greedily optimizes.

**Build pipeline**: One-command flow from model to bitstream: quantize, compile, simulate (Icarus Verilog), synthesize (Yosys), place-and-route (nextpnr), bitstream generation. Detects installed tools, reports pass/fail per stage.

> **Note:** Sequential mode currently supports Dense, Conv1D, and Conv2D layers (with fused ReLU). Other operations use combinational logic.

## Quick start

```bash
pip install numpy                     # only required dependency
pip install huggingface_hub safetensors  # optional: for HuggingFace models

# Compile GPT-2 to Verilog in one command
python -m w2s compile hf://openai-community/gpt2 --mode sequential --bits 8

# Or run the examples
python examples/xor_demo.py           # simplest demo
python examples/mnist_cnn_demo.py     # CNN compiled to silicon
python examples/sequential_demo.py    # sequential mode comparison
python examples/real_model_demo.py    # GPT-2 compiled to silicon
```

## CLI

```bash
python -m w2s compile model.onnx --mode sequential --bits 8
python -m w2s estimate model.onnx --mode both
python -m w2s info model.onnx
python -m w2s testbench model.onnx --vectors 8 --vcd
```

### HuggingFace direct import

Compile any supported HuggingFace model by ID — no ONNX export needed:

```bash
python -m w2s compile hf://openai-community/gpt2 --mode sequential --bits 8
python -m w2s info hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Supported architectures: GPT-2, Llama, Mistral, Qwen2, Phi, Gemma.

### Auto-fit

Automatically find the best quantization, sparsity, and mode to fit a target device:

```bash
python -m w2s autofit model.onnx --device ecp5-25k
python -m w2s autofit hf://openai-community/gpt2 --device ice40up5k
```

Runs per-layer sensitivity analysis, then greedily downgrades the least-sensitive layers to int4 and applies sparsity until the design fits.

### End-to-end build

One command: quantize, compile, simulate, synthesize, route, bitstream:

```bash
python -m w2s build model.onnx --device ice40up5k
python -m w2s build hf://openai-community/gpt2 --device ecp5-25k --mode sequential
```

Detects installed tools (Yosys, nextpnr, Icarus Verilog) and runs each stage with clear pass/fail reporting. Skips stages when tools are missing.

### Mixed precision

Assign different bit widths per layer to cut area where full precision isn't needed:

```bash
python -m w2s compile model.onnx --bits 8 --bits-map "attn_proj=4,classifier=16"
```

### FPGA targeting

Generate builds for real hardware (iCE40, ECP5) with Makefiles and pin constraints:

```bash
python -m w2s compile model.onnx --target fpga --device ecp5-25k --mode sequential
python -m w2s estimate model.onnx --target fpga --device ice40up5k
```

## Load a HuggingFace model

```python
from w2s.importers.hf_import import load_hf
from w2s.quantize import quantize_graph
from w2s.graph import compile_graph

graph = load_hf("openai-community/gpt2", blocks=[0])
quantize_graph(graph, {"token_embed": calibration_data})
compile_graph(graph, "output/", mode="sequential")
```

## Auto-fit to a device

```python
from w2s.importers.hf_import import load_hf
from w2s.autofit import autofit_fpga
from w2s.fpga import ECP5_25K

graph = load_hf("openai-community/gpt2", blocks=[0])
calib = {"token_embed": np.random.randn(4, 768).astype(np.float32)}
result = autofit_fpga(graph, calib, ECP5_25K)
print(result)  # mode, bits_map, sparsity, area estimate, fit status
```

## End-to-end build pipeline

```python
from w2s.pipeline import build

result = build(graph, calib, output_dir="./build", target="ice40up5k")
print(result)  # per-stage pass/fail, timing, output files
```

## Build a model with the GraphBuilder API

```python
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.graph import compile_graph

gb = GraphBuilder("my_model")
x = gb.input("x", shape=(784,))
h = gb.dense(x, W1, b1, activation="relu", name="hidden")
y = gb.dense(h, W2, b2, name="output")
gb.output(y)
graph = gb.build()

quantize_graph(graph, {"x": calibration_data})
compile_graph(graph, "output/")
```

## Or import an ONNX model

```python
from w2s.importers.onnx_import import load_onnx
from w2s.quantize import quantize_graph
from w2s.graph import compile_graph

graph = load_onnx("model.onnx")
quantize_graph(graph, {"input": calibration_data})
compile_graph(graph, "output/")
```

## Sparsity-aware compilation

Pruned models get proportionally smaller circuits — zero weights generate no hardware:

```python
from w2s.sparsity import analyze_sparsity, prune_weights, enforce_structured_2_4

# Analyze existing sparsity
report = analyze_sparsity(graph)
print(report)  # per-layer sparsity %, eliminated multipliers

# Prune to 50% sparsity (halves area for dense/conv layers)
prune_weights(graph, target_sparsity=0.5)

# Or enforce NVIDIA-style 2:4 structured sparsity
enforce_structured_2_4(graph)
```

## Mixed precision

Different layers have different quantization sensitivity. Assign bit widths per layer:

```python
# int4 for attention (area-hungry), int8 for everything else, int16 for classifier
bits_map = {"attn_qkv": 4, "attn_out": 4, "classifier": 16}
quantize_graph(graph, calibration_data, config, bits_map=bits_map)
compile_graph(graph, "output/")
```

## FPGA resource estimation

Check if your model fits on real hardware before synthesis:

```python
from w2s.fpga import estimate_fpga, ICE40_UP5K, ECP5_25K

report = estimate_fpga(graph, ICE40_UP5K, mode="sequential")
print(report)  # LUT4s, BRAM, DSP slices, utilization %, fit status
```

## Testbench generation

Generate Verilog testbenches with golden vectors for bit-exact verification:

```python
from w2s.graph import generate_testbench, forward_int

outputs = forward_int(graph, test_inputs)
generate_testbench(graph, test_inputs, outputs, "output/", vcd=True)
# Run with: iverilog -o sim model.v model_tb.v && vvp sim
```

## What the generated Verilog looks like

Every weight is a literal constant. No RAM, no ROM, no bus. The model IS the circuit:

```verilog
    // --- neuron [0][3] ---
    wire signed [31:0] l0_acc_3 =
          (32'sd85  * l0_ext_3_0)  // W=+0.6694
        + (32'sd-12 * l0_ext_3_1)  // W=-0.0945
        + 32'sd1234;               // bias=+0.1530

    wire signed [63:0] l0_req_3 = l0_ext64_3 * 64'sd8345;
    wire signed [63:0] l0_sh_3  = l0_req_3 >>> 16;

    wire signed [7:0] l0_out_3 =
        (l0_sh_3 > 64'sd127) ? 8'sd127 :
        (l0_sh_3 < 64'sd0)   ? 8'sd0 :    // ReLU
        l0_sh_3[7:0];
```

`32'sd85 * input` becomes a hardwired shift-add circuit.

## Validated

- Yosys synthesis: **zero errors** on all generated designs
- GPT-2 block 0: **5.9M pre-trained weights** compiled in ~5 seconds
- Area estimator: **~99% accuracy** vs actual Yosys gate count on tested designs
- Sequential mode: significantly fewer gates than combinational for same network
- **168 automated tests** covering quantization, compilation, estimation, sparsity, FPGA targeting, mixed precision, testbench generation, HuggingFace import, auto-fit, and build pipeline
- Tested end-to-end: HuggingFace download → quantize → forward_int (with MHA) → sequential Verilog for real GPT-2

## Architecture

```
w2s/
  core.py              — IR: ComputeGraph, Operation, TensorWires
  emit.py              — Verilog emission helpers
  quantize.py          — int4/8/16 quantization engine with calibration + mixed precision
  graph.py             — Main compiler: graph -> Verilog + testbench generation
  wrapper.py           — Serial I/O wrapper + Tiny Tapeout interface
  estimate.py          — Sparsity-aware area/gate estimation and Tiny Tapeout fit check
  sparsity.py          — Sparsity analysis, pruning, structured 2:4 enforcement
  fpga.py              — FPGA resource estimation, build scripts, pin constraints
  autofit.py           — Auto-fit: sensitivity analysis + greedy search for target device
  pipeline.py          — End-to-end build: quantize → compile → simulate → synthesize → bitstream
  __main__.py          — CLI tool (python -m w2s)
  sequential/
    compile.py         — Sequential ROM+MAC compiler with $readmemh
  generators/
    dense.py           — Fully connected layers
    conv.py            — Conv1D, Conv2D
    activation.py      — ReLU, GELU, Sigmoid, Tanh, SiLU, Softmax
    norm.py            — LayerNorm, RMSNorm, BatchNorm
    attention.py       — Multi-head self-attention
    transformer.py     — GQA, SwiGLU, RoPE, KV cache
    embedding.py       — Token embedding (ROM)
    pooling.py         — MaxPool, AvgPool, GlobalAvgPool
    structural.py      — Add, Multiply, Reshape, Flatten, Concat
  importers/
    builder.py         — Fluent GraphBuilder API
    onnx_import.py     — ONNX model import
    hf_import.py       — HuggingFace model import (GPT-2, Llama, Mistral, etc.)
```

## Cheers to the future

It feels like the window to democratize the near-term benefits of artificial intelligence as it currently exists is closing.  I want to contribute as much as possible to a future where people can still own and make things, and not rely on a supplier for performant LLMs.

## License

MIT
