# sandcastles

**Compile neural network weights directly into synthesizable Verilog.**

```
Trained model (.safetensors, .onnx, numpy)
        |
  [ sandcastles ]        <-- this tool
        |
  Synthesizable Verilog (hardwired weights)
        |
   OpenLane / LibreLane  
        |
  GDS-II layout
        |
   Tiny Tapeout / fab    
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

Architecturally capable of compiling any modern LLM: DeepSeek, Llama, Qwen, Mistral, Gemma.

**Quantization**: int4, int8, int16. Symmetric or asymmetric. Per-tensor or per-channel.

**Import**: ONNX models, numpy arrays, or the fluent GraphBuilder API.

**Output**: Combinational core, sequential ROM+MAC, serial I/O wrapper, Tiny Tapeout interface.

**Compilation modes**:
- *Combinational* — every weight gets its own multiplier. Max speed, max area.
- *Sequential* — one multiplier + weight ROM + state machine. Minimal area, fits Tiny Tapeout.

## Quick start

```bash
pip install numpy     # only required dependency (onnx optional)

python examples/xor_demo.py           # simplest demo
python examples/xor_graph_demo.py     # same, using graph API
python examples/mnist_cnn_demo.py     # CNN compiled to silicon
python examples/sequential_demo.py    # sequential mode comparison
python examples/real_model_demo.py    # GPT-2 compiled to silicon
```

## CLI

```bash
python -m w2s compile model.onnx --mode sequential --bits 8
python -m w2s estimate model.onnx --mode both
python -m w2s info model.onnx
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
- GPT-2 block 0: **5.9M pre-trained weights** compiled in 5.1 seconds
- Area estimator: **99.3% accuracy** vs actual Yosys gate count
- Sequential mode: **46% fewer gates** than combinational for same network

## Architecture

```
w2s/
  core.py              — IR: ComputeGraph, Operation, TensorWires
  emit.py              — Verilog emission helpers
  quantize.py          — int4/8/16 quantization engine with calibration
  graph.py             — Main compiler: graph -> Verilog
  wrapper.py           — Serial I/O wrapper + Tiny Tapeout interface
  estimate.py          — Area/gate estimation and Tiny Tapeout fit check
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
```

## Why this matters

A hardwired inference chip: **one-time fabrication cost, zero ongoing cost, runs forever, no cloud, no data exfiltration, no subscription**. The model becomes a physical object you own.

The fabrication tools are getting cheaper ([Tiny Tapeout](https://tinytapeout.com) = $100, [Atomic Semi](https://atomicsemi.com/) = affordable fabs, [Atum Works](https://www.ycombinator.com/companies/atum-works) = 3D-printed chips). The open-source design tools exist (OpenLane, Yosys, Magic). The only missing piece was a compiler that turns model weights into synthesizable Verilog targeting the free toolchain.  Now it's yours!

## License

MIT
