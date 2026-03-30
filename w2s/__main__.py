"""
__main__.py -- CLI entry point for weights2silicon.

Usage:
    python -m w2s compile model.onnx --mode sequential --bits 8 --output output/
    python -m w2s estimate model.onnx --mode both
    python -m w2s info model.onnx
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from w2s import __version__

BANNER = "weights2silicon (w2s) — Compile neural networks to hardwired Verilog"


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def _load_model(model_path: str, name: str = None):
    """Load a model file and return a ComputeGraph.

    Supports:
      - .onnx files (requires onnx package)
      - hf:// URIs for HuggingFace models (requires huggingface_hub + safetensors)
      - HuggingFace model IDs containing '/' (e.g., "openai-community/gpt2")
    """
    if model_path.startswith("hf://"):
        # Explicit HF prefix — always use HF
        model_id = model_path.removeprefix("hf://")
        try:
            from w2s.importers.hf_import import load_hf
        except ImportError:
            print(
                "Error: HuggingFace loading requires huggingface_hub and safetensors.\n"
                "Install them with:  pip install huggingface_hub safetensors",
                file=sys.stderr,
            )
            sys.exit(1)
        return load_hf(model_id, name=name)

    elif "/" in model_path and not Path(model_path).suffix and not Path(model_path).exists():
        # Looks like a HF model ID and doesn't exist as a local file
        model_id = model_path
        try:
            from w2s.importers.hf_import import load_hf
        except ImportError:
            print(
                "Error: HuggingFace loading requires huggingface_hub and safetensors.\n"
                "Install them with:  pip install huggingface_hub safetensors",
                file=sys.stderr,
            )
            sys.exit(1)
        return load_hf(model_id, name=name)

    else:
        # Treat as local file path
        path = Path(model_path)

        if not path.exists():
            print(f"Error: file not found: {model_path}", file=sys.stderr)
            sys.exit(1)

        ext = path.suffix.lower()

        if ext == ".onnx":
            try:
                from w2s.importers.onnx_import import load_onnx
            except ImportError:
                print(
                    "Error: the 'onnx' package is required to load .onnx models.\n"
                    "Install it with:  pip install onnx",
                    file=sys.stderr,
                )
                sys.exit(1)
            return load_onnx(str(path), name=name)

        else:
            print(
                f"Error: unsupported model format '{ext}'.\n"
                f"Supported: .onnx, or HuggingFace model ID (hf://model-id)",
                file=sys.stderr,
            )
            sys.exit(1)


# ---------------------------------------------------------------------------
#  Parameter counting helpers
# ---------------------------------------------------------------------------

def _count_params(graph):
    """Count total float parameters across all operations."""
    total = 0
    for op in graph.operations:
        for w in op.weights.values():
            total += int(np.prod(w.shape))
    return total


def _count_q_params(graph):
    """Count total quantized parameters across all operations."""
    total = 0
    for op in graph.operations:
        for w in op.q_weights.values():
            total += int(np.prod(w.shape))
    return total


# ---------------------------------------------------------------------------
#  Command: compile
# ---------------------------------------------------------------------------

def _parse_bits_map(bits_map_str: str) -> dict:
    """Parse a bits-map string like 'layer1=4,layer2=16' into a dict."""
    if not bits_map_str:
        return None
    result = {}
    for pair in bits_map_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"Malformed bits-map entry '{pair}': expected 'name=bits' format"
            )
        name, val = pair.split("=", 1)
        try:
            result[name.strip()] = int(val.strip())
        except ValueError:
            raise ValueError(
                f"Invalid bit width in bits-map entry '{pair}': "
                f"'{val.strip()}' is not an integer"
            )
    return result if result else None


def cmd_compile(args):
    from w2s.core import QuantConfig, QuantScheme, QuantGranularity

    model_name = args.name or Path(args.model).stem
    print(BANNER)
    print()
    print(f"  Model : {args.model}")
    print(f"  Name  : {model_name}")
    print(f"  Bits  : {args.bits}")
    print(f"  Output: {args.output}")

    bits_map = _parse_bits_map(getattr(args, 'bits_map', None))
    if bits_map:
        print(f"  Mixed : {bits_map}")

    target = getattr(args, 'target', 'asic')
    if target == 'fpga':
        print(f"  Target: FPGA ({getattr(args, 'device', 'ice40up5k')})")
    print()

    # 1. Load
    print("[1/4] Loading model...")
    graph = _load_model(args.model, name=model_name)
    total_params = _count_params(graph)
    print(f"       {len(graph.operations)} operations, {total_params:,} parameters")

    # 2. Configure quantization
    print(f"[2/4] Quantizing to int{args.bits}...")
    config = QuantConfig(
        bits=args.bits,
        scheme=QuantScheme.SYMMETRIC,
        granularity=QuantGranularity.PER_TENSOR,
    )
    graph.quant_config = config

    # Generate random calibration data and warn the user
    calib_data = {}
    for inp_name in graph.input_names:
        shape = graph.input_shapes.get(inp_name, (1,))
        calib_shape = (4,) + tuple(shape)
        calib_data[inp_name] = np.random.randn(*calib_shape).astype(np.float32)

    print("       WARNING: using random calibration data.  For better accuracy,")
    print("       supply real calibration data from your dataset.")

    from w2s.quantize import quantize_graph
    graph = quantize_graph(graph, calib_data, config, bits_map=bits_map)

    q_params = _count_q_params(graph)
    print(f"       Quantized {q_params:,} parameters to int{args.bits}")

    # 2b. Sparsity report
    from w2s.sparsity import analyze_sparsity
    sp_report = analyze_sparsity(graph)
    if sp_report.overall_sparsity > 0.01:
        print(f"       Sparsity: {sp_report.overall_sparsity:.1%} "
              f"({sp_report.eliminated_multipliers:,} multipliers eliminated)")

    # 3. Select mode
    mode = args.mode
    if mode == "auto":
        mode = "sequential" if total_params > 50000 else "combinational"
        print(f"[3/4] Auto-selected mode: {mode} ({total_params:,} params)")
    else:
        print(f"[3/4] Mode: {mode}")

    # 4. Compile
    print("[4/4] Compiling to Verilog...")
    from w2s.graph import compile_graph
    output_path = compile_graph(graph, output_dir=args.output, mode=mode)

    print()
    print("  Done!")
    print(f"  Verilog : {output_path}")
    print(f"  Mode    : {mode}")
    print(f"  Params  : {q_params:,} (all hardwired as constants)")

    # FPGA build script
    if target == 'fpga':
        from w2s.fpga import generate_build_script, generate_constraints, DEVICES
        device_name = getattr(args, 'device', 'ice40up5k')
        device = DEVICES.get(device_name)
        if device:
            mk_path = generate_build_script(graph, device, args.output, mode)
            pcf_path = generate_constraints(graph, device, args.output, mode)
            print(f"  Makefile: {mk_path}")
            print(f"  Pins   : {pcf_path}")

    print()


# ---------------------------------------------------------------------------
#  Command: estimate
# ---------------------------------------------------------------------------

def cmd_estimate(args):
    print(BANNER)
    print()
    print(f"  Model: {args.model}")
    print(f"  Bits : {args.bits}")
    print()

    print("Loading model...")
    graph = _load_model(args.model)
    total_params = _count_params(graph)

    # Quantize the graph so the estimator has accurate weight information
    from w2s.core import QuantConfig, QuantScheme, QuantGranularity
    config = QuantConfig(
        bits=args.bits,
        scheme=QuantScheme.SYMMETRIC,
        granularity=QuantGranularity.PER_TENSOR,
    )
    graph.quant_config = config

    calib_data = {}
    for inp_name in graph.input_names:
        shape = graph.input_shapes.get(inp_name, (1,))
        calib_shape = (4,) + tuple(shape)
        calib_data[inp_name] = np.random.randn(*calib_shape).astype(np.float32)

    from w2s.quantize import quantize_graph
    graph = quantize_graph(graph, calib_data, config)

    modes = [args.mode] if args.mode != "both" else ["combinational", "sequential"]

    # Sparsity report
    from w2s.sparsity import analyze_sparsity
    sp_report = analyze_sparsity(graph)
    if sp_report.overall_sparsity > 0.001:
        print(sp_report)
        print()

    # ASIC estimate
    from w2s.estimate import estimate as run_estimate
    for mode in modes:
        print(f"\n--- ASIC {mode} ---")
        report = run_estimate(graph, mode=mode)
        print(report)

    # FPGA estimate (if requested)
    target = getattr(args, 'target', 'asic')
    if target in ('fpga', 'both'):
        from w2s.fpga import estimate_fpga, DEVICES
        device_name = getattr(args, 'device', 'ice40up5k')
        device = DEVICES.get(device_name)
        if device:
            for mode in modes:
                print(f"\n--- FPGA {device.name} {mode} ---")
                fpga_report = estimate_fpga(graph, device, mode)
                print(fpga_report)


# ---------------------------------------------------------------------------
#  Command: info
# ---------------------------------------------------------------------------

def cmd_testbench(args):
    """Generate a testbench with golden vectors from the quantized model."""
    print(BANNER)
    print()
    print(f"  Model : {args.model}")
    print(f"  Bits  : {args.bits}")
    print(f"  Output: {args.output}")
    print()

    from w2s.core import QuantConfig, QuantScheme, QuantGranularity

    model_name = args.name or Path(args.model).stem

    # 1. Load
    print("[1/3] Loading model...")
    graph = _load_model(args.model, name=model_name)
    total_params = _count_params(graph)
    print(f"       {len(graph.operations)} operations, {total_params:,} parameters")

    # 2. Quantize
    print(f"[2/3] Quantizing to int{args.bits}...")
    config = QuantConfig(
        bits=args.bits,
        scheme=QuantScheme.SYMMETRIC,
        granularity=QuantGranularity.PER_TENSOR,
    )
    graph.quant_config = config

    calib_data = {}
    for inp_name in graph.input_names:
        shape = graph.input_shapes.get(inp_name, (1,))
        calib_shape = (4,) + tuple(shape)
        calib_data[inp_name] = np.random.randn(*calib_shape).astype(np.float32)

    from w2s.quantize import quantize_graph
    graph = quantize_graph(graph, calib_data, config)

    # 3. Generate golden vectors and testbench
    print("[3/3] Generating testbench...")
    n_vectors = getattr(args, 'vectors', 4)
    vcd = getattr(args, 'vcd', False)
    tolerance = getattr(args, 'tolerance', 0)

    # Generate test inputs (random quantized values)
    test_inputs = {}
    bits = args.bits
    qmax = 2 ** (bits - 1) - 1
    for inp_name in graph.input_names:
        shape = graph.input_shapes.get(inp_name, (1,))
        numel = 1
        for s in shape:
            numel *= s
        test_inputs[inp_name] = np.random.randint(
            -qmax, qmax + 1, size=(n_vectors, numel)).astype(np.int64)

    # Run integer forward pass for golden outputs
    from w2s.graph import forward_int, generate_testbench

    # Run forward pass per vector
    all_outputs = {}
    for t in range(n_vectors):
        single_input = {}
        for inp_name in graph.input_names:
            vec = test_inputs[inp_name][t].astype(np.float64)
            scale = graph.tensor_scales.get(inp_name, 1.0)
            single_input[inp_name] = vec / scale if scale != 0 else vec
        outputs = forward_int(graph, single_input)
        for out_name, val in outputs.items():
            if out_name not in all_outputs:
                all_outputs[out_name] = []
            all_outputs[out_name].append(val.flatten())

    expected_outputs = {}
    for out_name, vecs in all_outputs.items():
        expected_outputs[out_name] = np.stack(vecs, axis=0)

    mode = getattr(args, 'mode', 'combinational')

    tb_path = None
    if mode in ("combinational", "both"):
        tb_path = generate_testbench(
            graph, test_inputs, expected_outputs,
            output_dir=args.output, vcd=vcd, tolerance=tolerance,
        )

    if mode in ("sequential", "both"):
        from w2s.graph import generate_sequential_testbench
        seq_tb = generate_sequential_testbench(
            graph, test_inputs, expected_outputs,
            output_dir=args.output, vcd=vcd, tolerance=tolerance,
        )
        print(f"  Sequential TB: {seq_tb}")
        if tb_path is None:
            tb_path = seq_tb

    print()
    print("  Done!")
    print(f"  Testbench: {tb_path}")
    print(f"  Vectors  : {n_vectors}")
    if vcd:
        print(f"  VCD dump : enabled")
    print()


def cmd_info(args):
    print(BANNER)
    print()

    print("Loading model...")
    graph = _load_model(args.model)
    total_params = _count_params(graph)

    print()
    print(f"  Model      : {graph.name}")
    print(f"  Operations : {len(graph.operations)}")
    print(f"  Inputs     : {', '.join(graph.input_names)}")
    print(f"  Outputs    : {', '.join(graph.output_names)}")
    print(f"  Parameters : {total_params:,}")
    print(f"  Est. memory: {total_params * 4 / 1024:.1f} KB (float32)")
    print()

    # Input shapes
    if graph.input_shapes:
        print("  Input shapes:")
        for name, shape in graph.input_shapes.items():
            print(f"    {name}: {shape}")
        print()

    # Layer table
    print("  Layers:")
    print(f"  {'Name':<30} {'Type':<16} {'Params':>10}  Details")
    print(f"  {'-'*30} {'-'*16} {'-'*10}  {'-'*30}")
    for op in graph.operations:
        n_params = sum(int(np.prod(w.shape)) for w in op.weights.values())
        param_str = f"{n_params:,}" if n_params > 0 else "-"

        details = []
        for wname, warr in op.weights.items():
            details.append(f"{wname}{list(warr.shape)}")
        if op.attrs:
            for k, v in op.attrs.items():
                details.append(f"{k}={v}")
        detail_str = ", ".join(details) if details else ""

        print(f"  {op.name:<30} {op.op_type.value:<16} {param_str:>10}  {detail_str}")

    print()
    print(f"  Total: {total_params:,} parameters")
    print()


# ---------------------------------------------------------------------------
#  Command: autofit
# ---------------------------------------------------------------------------

def cmd_autofit(args):
    print(BANNER)
    print()
    print(f"  Model  : {args.model}")
    print(f"  Device : {args.device}")
    print()

    model_name = args.name or Path(args.model).stem
    print("Loading model...")
    graph = _load_model(args.model, name=model_name)
    total_params = _count_params(graph)
    print(f"  {len(graph.operations)} operations, {total_params:,} parameters")
    print()

    # Generate calibration data
    calib_data = {}
    for inp_name in graph.input_names:
        shape = graph.input_shapes.get(inp_name, (1,))
        calib_shape = (4,) + tuple(shape)
        calib_data[inp_name] = np.random.randn(*calib_shape).astype(np.float32)

    # Run auto-fit
    from w2s.fpga import DEVICES
    from w2s.autofit import autofit_fpga, analyze_sensitivity

    device = DEVICES.get(args.device)
    if device is None:
        print(f"Error: unknown device '{args.device}'", file=sys.stderr)
        sys.exit(1)

    print(f"Running sensitivity analysis...")
    sensitivity = analyze_sensitivity(graph, calib_data)
    print(sensitivity)
    print()

    print(f"Searching for best configuration to fit {device.name} "
          f"({device.lut4s:,} LUT4s)...")
    result = autofit_fpga(
        graph, calib_data, device,
        max_sparsity=getattr(args, 'max_sparsity', 0.75),
    )
    print()
    print(result)
    print()

    if result.fits and result.bits_map:
        print("  To compile with this configuration:")
        bm = ",".join(f"{k}={v}" for k, v in result.bits_map.items() if v != result.bits)
        print(f"    python -m w2s compile {args.model} --bits {result.bits} "
              f"--mode {result.mode} --bits-map \"{bm}\"")
        print()


# ---------------------------------------------------------------------------
#  Command: build
# ---------------------------------------------------------------------------

def cmd_build(args):
    print(BANNER)
    print()

    model_name = args.name or Path(args.model).stem
    print(f"  Model  : {args.model}")
    print(f"  Device : {args.device}")
    print(f"  Output : {args.output}")
    print()

    # Detect tools
    from w2s.pipeline import detect_tools, build
    tools = detect_tools()
    print("  Tools detected:")
    for name, status in tools.items():
        if status.available:
            print(f"    {name:<20} {status.version}")
    missing = [t.name for t in tools.values() if not t.available]
    if missing:
        print(f"    Missing: {', '.join(missing)}")
    print()

    # Load model
    print("Loading model...")
    graph = _load_model(args.model, name=model_name)
    total_params = _count_params(graph)
    print(f"  {len(graph.operations)} operations, {total_params:,} parameters")
    print()

    # Generate calibration data
    calib_data = {}
    for inp_name in graph.input_names:
        shape = graph.input_shapes.get(inp_name, (1,))
        calib_shape = (4,) + tuple(shape)
        calib_data[inp_name] = np.random.randn(*calib_shape).astype(np.float32)

    # Select mode
    mode = args.mode
    if mode == "auto":
        mode = "sequential" if total_params > 50000 else "combinational"
        print(f"  Auto-selected mode: {mode}")

    bits_map = _parse_bits_map(getattr(args, 'bits_map', None))

    # Run pipeline
    result = build(
        graph, calib_data,
        output_dir=args.output,
        mode=mode,
        bits=args.bits,
        bits_map=bits_map,
        target=args.device,
        simulate=not args.no_simulate,
        synthesize=not args.no_synthesize,
        route=not args.no_synthesize,
        bitstream=not args.no_synthesize,
    )

    print(result)


# ---------------------------------------------------------------------------
#  Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="w2s",
        description=BANNER,
    )
    parser.add_argument(
        "--version", action="version", version=f"w2s {__version__}",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print full tracebacks on error",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- compile ---
    p_compile = subparsers.add_parser(
        "compile",
        help="Compile a neural network model to synthesizable Verilog",
    )
    p_compile.add_argument(
        "model",
        help="Path to model file (.onnx or .safetensors)",
    )
    p_compile.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    p_compile.add_argument(
        "--mode", "-m",
        choices=["combinational", "sequential", "auto"],
        default="auto",
        help="Compilation mode (default: auto-select based on model size)",
    )
    p_compile.add_argument(
        "--bits", "-b",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Quantization bit width (default: 8)",
    )
    p_compile.add_argument(
        "--name", "-n",
        default=None,
        help="Verilog module name (default: derived from filename)",
    )
    p_compile.add_argument(
        "--bits-map",
        default=None,
        help="Mixed-precision per-layer bit widths (e.g., 'hidden=4,output=16')",
    )
    p_compile.add_argument(
        "--target", "-t",
        choices=["asic", "fpga"],
        default="asic",
        help="Target platform (default: asic)",
    )
    p_compile.add_argument(
        "--device",
        choices=["ice40up5k", "ice40hx8k", "ecp5-25k", "ecp5-85k"],
        default="ice40up5k",
        help="FPGA device (only used with --target fpga, default: ice40up5k)",
    )

    # --- estimate ---
    p_estimate = subparsers.add_parser(
        "estimate",
        help="Estimate area and resource usage without full compilation",
    )
    p_estimate.add_argument(
        "model",
        help="Path to model file (.onnx or .safetensors)",
    )
    p_estimate.add_argument(
        "--mode", "-m",
        choices=["combinational", "sequential", "both"],
        default="both",
        help="Estimation mode (default: both)",
    )
    p_estimate.add_argument(
        "--bits", "-b",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Quantization bit width (default: 8)",
    )
    p_estimate.add_argument(
        "--target", "-t",
        choices=["asic", "fpga", "both"],
        default="asic",
        help="Target platform for estimation (default: asic)",
    )
    p_estimate.add_argument(
        "--device",
        choices=["ice40up5k", "ice40hx8k", "ecp5-25k", "ecp5-85k"],
        default="ice40up5k",
        help="FPGA device (only used with --target fpga/both, default: ice40up5k)",
    )

    # --- testbench ---
    p_testbench = subparsers.add_parser(
        "testbench",
        help="Generate a Verilog testbench with golden vectors",
    )
    p_testbench.add_argument(
        "model",
        help="Path to model file (.onnx or .safetensors)",
    )
    p_testbench.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    p_testbench.add_argument(
        "--bits", "-b",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Quantization bit width (default: 8)",
    )
    p_testbench.add_argument(
        "--name", "-n",
        default=None,
        help="Verilog module name (default: derived from filename)",
    )
    p_testbench.add_argument(
        "--vectors", "-v",
        type=int,
        default=4,
        help="Number of test vectors to generate (default: 4)",
    )
    p_testbench.add_argument(
        "--vcd",
        action="store_true",
        default=False,
        help="Include VCD waveform dump in testbench",
    )
    p_testbench.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Allowed output tolerance in LSBs (default: 0 = exact match)",
    )
    p_testbench.add_argument(
        "--mode", "-m",
        choices=["combinational", "sequential", "both"],
        default="combinational",
        help="Generate testbench for which mode (default: combinational)",
    )

    # --- autofit ---
    p_autofit = subparsers.add_parser(
        "autofit",
        help="Automatically find the best quantization to fit a target device",
    )
    p_autofit.add_argument(
        "model",
        help="Path to model file, ONNX, or HuggingFace model ID (hf://model-id)",
    )
    p_autofit.add_argument(
        "--device",
        choices=["ice40up5k", "ice40hx8k", "ecp5-25k", "ecp5-85k"],
        default="ice40up5k",
        help="Target FPGA device (default: ice40up5k)",
    )
    p_autofit.add_argument(
        "--name", "-n",
        default=None,
        help="Verilog module name (default: derived from filename)",
    )
    p_autofit.add_argument(
        "--max-sparsity",
        type=float,
        default=0.75,
        help="Maximum sparsity level to try (default: 0.75)",
    )

    # --- build ---
    p_build = subparsers.add_parser(
        "build",
        help="End-to-end pipeline: quantize → compile → simulate → synthesize → bitstream",
    )
    p_build.add_argument(
        "model",
        help="Path to model file, ONNX, or HuggingFace model ID (hf://model-id)",
    )
    p_build.add_argument(
        "--output", "-o",
        default="./build",
        help="Output directory (default: ./build)",
    )
    p_build.add_argument(
        "--mode", "-m",
        choices=["combinational", "sequential", "auto"],
        default="auto",
        help="Compilation mode (default: auto)",
    )
    p_build.add_argument(
        "--bits", "-b",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Quantization bit width (default: 8)",
    )
    p_build.add_argument(
        "--name", "-n",
        default=None,
        help="Verilog module name (default: derived from filename)",
    )
    p_build.add_argument(
        "--device",
        choices=["ice40up5k", "ice40hx8k", "ecp5-25k", "ecp5-85k"],
        default="ice40up5k",
        help="Target FPGA device (default: ice40up5k)",
    )
    p_build.add_argument(
        "--bits-map",
        default=None,
        help="Mixed-precision per-layer bit widths (e.g., 'hidden=4,output=16')",
    )
    p_build.add_argument(
        "--no-simulate",
        action="store_true",
        default=False,
        help="Skip simulation stage",
    )
    p_build.add_argument(
        "--no-synthesize",
        action="store_true",
        default=False,
        help="Skip synthesis and later stages",
    )

    # --- info ---
    p_info = subparsers.add_parser(
        "info",
        help="Show model information (layers, parameters, shapes)",
    )
    p_info.add_argument(
        "model",
        help="Path to model file (.onnx or .safetensors)",
    )

    return parser


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "compile":
            cmd_compile(args)
        elif args.command == "estimate":
            cmd_estimate(args)
        elif args.command == "testbench":
            cmd_testbench(args)
        elif args.command == "autofit":
            cmd_autofit(args)
        elif args.command == "build":
            cmd_build(args)
        elif args.command == "info":
            cmd_info(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
