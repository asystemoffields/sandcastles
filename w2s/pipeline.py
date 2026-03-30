"""
pipeline.py — End-to-end build pipeline.

Orchestrates the full flow from model to bitstream:
  1. Load model (ONNX or HuggingFace)
  2. Quantize (with optional auto-fit)
  3. Compile to Verilog
  4. Generate testbench
  5. Simulate (iverilog + vvp)
  6. Synthesize (Yosys)
  7. Place and route (nextpnr)
  8. Generate bitstream (icepack / ecppack)

Each stage reports pass/fail with clear error messages.
Missing tools are detected upfront.
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from w2s.core import ComputeGraph, QuantConfig, QuantScheme


# ---------------------------------------------------------------------------
#  Tool detection
# ---------------------------------------------------------------------------

@dataclass
class ToolStatus:
    """Status of an external tool."""
    name: str
    path: Optional[str]
    available: bool
    version: str = ""


def detect_tools() -> Dict[str, ToolStatus]:
    """Detect which FPGA/simulation tools are installed."""
    tools = {}
    for name in ["yosys", "nextpnr-ice40", "nextpnr-ecp5",
                  "icepack", "iceprog", "ecppack",
                  "iverilog", "vvp"]:
        path = shutil.which(name)
        version = ""
        if path:
            try:
                result = subprocess.run(
                    [name, "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                version = result.stdout.strip().split("\n")[0]
                if not version:
                    version = result.stderr.strip().split("\n")[0]
            except Exception:
                version = "installed"
        tools[name] = ToolStatus(
            name=name,
            path=path,
            available=path is not None,
            version=version,
        )
    return tools


# ---------------------------------------------------------------------------
#  Pipeline stages
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    name: str
    passed: bool
    duration_s: float
    output_files: List[str] = field(default_factory=list)
    message: str = ""
    log: str = ""


@dataclass
class PipelineResult:
    """Result of the full pipeline run."""
    stages: List[StageResult] = field(default_factory=list)
    success: bool = False
    output_dir: str = ""

    def __str__(self) -> str:
        lines = ["", "=== Build Pipeline Results ===", ""]
        max_name = max((len(s.name) for s in self.stages), default=10)
        for s in self.stages:
            status = "PASS" if s.passed else "FAIL"
            lines.append(f"  [{status}] {s.name:<{max_name}}  {s.duration_s:.1f}s  {s.message}")
        lines.append("")
        if self.success:
            lines.append("  BUILD SUCCEEDED")
        else:
            failed = [s for s in self.stages if not s.passed]
            if failed:
                lines.append(f"  BUILD FAILED at stage: {failed[0].name}")
                if failed[0].log:
                    lines.append("")
                    lines.append("  Error log:")
                    for line in failed[0].log.split("\n")[:20]:
                        lines.append(f"    {line}")
        lines.append(f"  Output: {self.output_dir}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def build(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
    output_dir: str = "./build",
    mode: str = "sequential",
    bits: int = 8,
    bits_map: Optional[Dict[str, int]] = None,
    target: str = "ice40up5k",
    simulate: bool = True,
    synthesize: bool = True,
    route: bool = True,
    bitstream: bool = True,
) -> PipelineResult:
    """
    Run the full build pipeline.

    Args:
        graph:            ComputeGraph with float weights.
        calibration_data: Calibration data for quantization.
        output_dir:       Where to write all output files.
        mode:             "combinational" or "sequential".
        bits:             Default quantization bit width.
        bits_map:         Per-layer bit width overrides.
        target:           Target device name (e.g., "ice40up5k", "ecp5-25k").
        simulate:         Run Icarus Verilog simulation.
        synthesize:       Run Yosys synthesis.
        route:            Run nextpnr place-and-route.
        bitstream:        Generate bitstream.

    Returns:
        PipelineResult with per-stage pass/fail.
    """
    from w2s.autofit import _deep_copy_graph
    graph = _deep_copy_graph(graph)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = PipelineResult(output_dir=str(out))

    # Check tools upfront
    tools = detect_tools()

    # Resolve device
    from w2s.fpga import DEVICES
    device = DEVICES.get(target)
    if device is None:
        result.stages.append(StageResult(
            name="Setup",
            passed=False,
            duration_s=0,
            message=f"Unknown device '{target}'. Available: {', '.join(DEVICES.keys())}",
        ))
        return result

    # Stage 1: Quantize
    stage = _stage_quantize(graph, calibration_data, bits, bits_map)
    result.stages.append(stage)
    if not stage.passed:
        return result

    # Stage 2: Compile
    stage = _stage_compile(graph, str(out), mode)
    result.stages.append(stage)
    if not stage.passed:
        return result
    verilog_files = stage.output_files

    # Stage 3: Testbench
    stage = _stage_testbench(graph, calibration_data, str(out), bits)
    result.stages.append(stage)
    if not stage.passed:
        return result
    tb_files = stage.output_files

    # Stage 4: Simulate (optional, requires iverilog)
    if simulate:
        if tools["iverilog"].available and tools["vvp"].available:
            stage = _stage_simulate(verilog_files, tb_files, str(out))
            result.stages.append(stage)
            if not stage.passed:
                return result
        else:
            result.stages.append(StageResult(
                name="Simulate",
                passed=True,
                duration_s=0,
                message="SKIPPED (iverilog not found)",
            ))

    # Stage 5: Synthesize (optional, requires yosys)
    if synthesize:
        if tools["yosys"].available:
            stage = _stage_synthesize(verilog_files, str(out), device, mode)
            result.stages.append(stage)
            if not stage.passed:
                return result
        else:
            result.stages.append(StageResult(
                name="Synthesize",
                passed=True,
                duration_s=0,
                message="SKIPPED (yosys not found)",
            ))

    # Stage 6: Place and route (optional)
    if route and synthesize:
        pnr_tool = f"nextpnr-{device.family}"
        if tools.get(pnr_tool, ToolStatus("", None, False)).available:
            stage = _stage_route(str(out), device, mode, graph.name)
            result.stages.append(stage)
            if not stage.passed:
                return result
        else:
            result.stages.append(StageResult(
                name="Place & Route",
                passed=True,
                duration_s=0,
                message=f"SKIPPED ({pnr_tool} not found)",
            ))

    # Stage 7: Bitstream (optional)
    if bitstream and route and synthesize:
        pack_tool = "icepack" if device.family == "ice40" else "ecppack"
        if tools.get(pack_tool, ToolStatus("", None, False)).available:
            stage = _stage_bitstream(str(out), device, mode, graph.name)
            result.stages.append(stage)
            if not stage.passed:
                return result
        else:
            result.stages.append(StageResult(
                name="Bitstream",
                passed=True,
                duration_s=0,
                message=f"SKIPPED ({pack_tool} not found)",
            ))

    # Generate build script for manual re-runs
    from w2s.fpga import generate_build_script, generate_constraints
    generate_build_script(graph, device, str(out), mode)
    generate_constraints(graph, device, str(out), mode)

    result.success = all(s.passed for s in result.stages)
    return result


# ---------------------------------------------------------------------------
#  Stage implementations
# ---------------------------------------------------------------------------

def _stage_quantize(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
    bits: int,
    bits_map: Optional[Dict[str, int]],
) -> StageResult:
    """Stage 1: Quantize the model."""
    t0 = time.time()
    try:
        config = QuantConfig(bits=bits, scheme=QuantScheme.SYMMETRIC)
        graph.quant_config = config
        from w2s.quantize import quantize_graph
        quantize_graph(graph, calibration_data, config, bits_map=bits_map)

        n_params = sum(
            int(np.prod(w.shape))
            for op in graph.operations
            for w in op.q_weights.values()
        )
        return StageResult(
            name="Quantize",
            passed=True,
            duration_s=time.time() - t0,
            message=f"{n_params:,} params → int{bits}",
        )
    except Exception as e:
        return StageResult(
            name="Quantize",
            passed=False,
            duration_s=time.time() - t0,
            message=str(e),
            log=str(e),
        )


def _stage_compile(
    graph: ComputeGraph,
    output_dir: str,
    mode: str,
) -> StageResult:
    """Stage 2: Compile to Verilog."""
    t0 = time.time()
    try:
        from w2s.graph import compile_graph
        v_path = compile_graph(graph, output_dir=output_dir, mode=mode)

        # Collect all output files (Verilog + hex)
        out = Path(output_dir)
        verilog_files = [str(p) for p in out.glob("*.v")]
        hex_files = [str(p) for p in out.glob("*.hex")]

        v_lines = Path(v_path).read_text().count("\n")
        return StageResult(
            name="Compile",
            passed=True,
            duration_s=time.time() - t0,
            output_files=verilog_files,
            message=f"{v_lines:,} lines of Verilog, {len(hex_files)} hex files",
        )
    except Exception as e:
        return StageResult(
            name="Compile",
            passed=False,
            duration_s=time.time() - t0,
            message=str(e),
            log=str(e),
        )


def _stage_testbench(
    graph: ComputeGraph,
    calibration_data: Dict[str, np.ndarray],
    output_dir: str,
    bits: int,
) -> StageResult:
    """Stage 3: Generate testbench with golden vectors."""
    t0 = time.time()
    try:
        from w2s.graph import generate_testbench, forward_int

        qmax = 2 ** (bits - 1) - 1
        n_vectors = min(4, max(1, calibration_data[
            list(calibration_data.keys())[0]].shape[0]))

        # Generate test inputs
        test_inputs = {}
        for inp_name in graph.input_names:
            shape = graph.input_shapes.get(inp_name, (1,))
            numel = 1
            for s in shape:
                numel *= s
            test_inputs[inp_name] = np.random.randint(
                -qmax, qmax + 1, size=(n_vectors, numel)).astype(np.int64)

        # Generate golden outputs
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

        expected_outputs = {
            name: np.stack(vecs, axis=0)
            for name, vecs in all_outputs.items()
        }

        tb_path = generate_testbench(
            graph, test_inputs, expected_outputs,
            output_dir=output_dir, vcd=True,
        )

        return StageResult(
            name="Testbench",
            passed=True,
            duration_s=time.time() - t0,
            output_files=[tb_path],
            message=f"{n_vectors} test vectors",
        )
    except Exception as e:
        return StageResult(
            name="Testbench",
            passed=False,
            duration_s=time.time() - t0,
            message=str(e),
            log=str(e),
        )


def _stage_simulate(
    verilog_files: List[str],
    tb_files: List[str],
    output_dir: str,
) -> StageResult:
    """Stage 4: Simulate with Icarus Verilog."""
    t0 = time.time()
    try:
        out = Path(output_dir)
        sim_bin = str(out / "sim_out")
        all_files = verilog_files + tb_files

        # Compile
        cmd_compile = ["iverilog", "-o", sim_bin] + all_files
        result = subprocess.run(
            cmd_compile, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return StageResult(
                name="Simulate",
                passed=False,
                duration_s=time.time() - t0,
                message="iverilog compilation failed",
                log=result.stderr,
            )

        # Run
        cmd_run = ["vvp", sim_bin]
        result = subprocess.run(
            cmd_run, capture_output=True, text=True, timeout=300,
            cwd=str(out),
        )

        output = result.stdout + result.stderr
        passed = "PASS \u2014" in output or "PASS --" in output
        failed = "FAILED \u2014" in output or "FAILED --" in output
        passed = passed and not failed

        return StageResult(
            name="Simulate",
            passed=passed,
            duration_s=time.time() - t0,
            message="PASS" if passed else "Simulation mismatch",
            log=output if not passed else "",
        )
    except subprocess.TimeoutExpired:
        return StageResult(
            name="Simulate",
            passed=False,
            duration_s=time.time() - t0,
            message="Simulation timed out",
        )
    except Exception as e:
        return StageResult(
            name="Simulate",
            passed=False,
            duration_s=time.time() - t0,
            message=str(e),
            log=str(e),
        )


def _stage_synthesize(
    verilog_files: List[str],
    output_dir: str,
    device,
    mode: str,
) -> StageResult:
    """Stage 5: Synthesize with Yosys."""
    t0 = time.time()
    try:
        out = Path(output_dir)
        json_out = str(out / "design.json")
        synth_cmd = f"synth_{device.family}"

        # Determine top module name
        if mode == "sequential":
            # Find the _seq.v file to get the module name
            seq_files = [f for f in verilog_files if f.endswith("_seq.v")]
            if seq_files:
                top = Path(seq_files[0]).stem
            else:
                top = Path(verilog_files[0]).stem
        else:
            # Use the main .v file (not _tb, not _serial, not _seq)
            main_files = [
                f for f in verilog_files
                if not any(s in f for s in ["_tb", "_serial", "_seq", "tt_um_"])
            ]
            top = Path(main_files[0]).stem if main_files else Path(verilog_files[0]).stem

        # Filter to non-testbench Verilog files
        src_files = [f for f in verilog_files if "_tb" not in f]

        yosys_script = f"{synth_cmd} -top {top} -json {json_out}"
        cmd = ["yosys", "-p", yosys_script] + src_files

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )

        if result.returncode != 0:
            return StageResult(
                name="Synthesize",
                passed=False,
                duration_s=time.time() - t0,
                message="Yosys synthesis failed",
                log=result.stderr[-2000:] if result.stderr else result.stdout[-2000:],
            )

        # Extract resource usage from Yosys output
        message = "synthesis complete"
        for line in result.stdout.split("\n"):
            if "Number of cells:" in line or "Number of wires:" in line:
                message = line.strip()
                break

        return StageResult(
            name="Synthesize",
            passed=True,
            duration_s=time.time() - t0,
            output_files=[json_out],
            message=message,
        )
    except subprocess.TimeoutExpired:
        return StageResult(
            name="Synthesize",
            passed=False,
            duration_s=time.time() - t0,
            message="Yosys timed out (>10 min)",
        )
    except Exception as e:
        return StageResult(
            name="Synthesize",
            passed=False,
            duration_s=time.time() - t0,
            message=str(e),
            log=str(e),
        )


def _stage_route(
    output_dir: str,
    device,
    mode: str,
    graph_name: str,
) -> StageResult:
    """Stage 6: Place and route with nextpnr."""
    t0 = time.time()
    try:
        out = Path(output_dir)
        json_in = str(out / "design.json")

        if device.family == "ice40":
            if "hx8k" in device.name.lower():
                dev_flag = "hx8k"
            elif "up5k" in device.name.lower():
                dev_flag = "up5k"
            else:
                dev_flag = "lp8k"

            asc_out = str(out / "design.asc")
            empty_pcf = str(out / "empty.pcf")
            Path(empty_pcf).write_text("# empty constraints for validation\n")
            cmd = [
                "nextpnr-ice40",
                f"--{dev_flag}",
                "--package", device.package or "sg48",
                "--json", json_in,
                "--asc", asc_out,
                "--pcf", empty_pcf,
            ]
            output_file = asc_out

        elif device.family == "ecp5":
            if "25k" in device.name.lower():
                dev_flag = "25k"
            elif "85k" in device.name.lower():
                dev_flag = "85k"
            else:
                dev_flag = "45k"

            config_out = str(out / "design.config")
            cmd = [
                "nextpnr-ecp5",
                f"--{dev_flag}",
                "--package", device.package or "CABGA381",
                "--json", json_in,
                "--textcfg", config_out,
            ]
            output_file = config_out
        else:
            return StageResult(
                name="Place & Route",
                passed=False,
                duration_s=0,
                message=f"Unsupported device family: {device.family}",
            )

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )

        if result.returncode != 0:
            return StageResult(
                name="Place & Route",
                passed=False,
                duration_s=time.time() - t0,
                message="nextpnr failed",
                log=result.stderr[-2000:] if result.stderr else result.stdout[-2000:],
            )

        # Extract timing from nextpnr output
        message = "routing complete"
        for line in (result.stdout + result.stderr).split("\n"):
            if "Max frequency" in line or "Fmax" in line:
                message = line.strip()
                break

        return StageResult(
            name="Place & Route",
            passed=True,
            duration_s=time.time() - t0,
            output_files=[output_file],
            message=message,
        )
    except subprocess.TimeoutExpired:
        return StageResult(
            name="Place & Route",
            passed=False,
            duration_s=time.time() - t0,
            message="nextpnr timed out (>10 min)",
        )
    except Exception as e:
        return StageResult(
            name="Place & Route",
            passed=False,
            duration_s=time.time() - t0,
            message=str(e),
            log=str(e),
        )


def _stage_bitstream(
    output_dir: str,
    device,
    mode: str,
    graph_name: str,
) -> StageResult:
    """Stage 7: Generate bitstream."""
    t0 = time.time()
    try:
        out = Path(output_dir)

        if device.family == "ice40":
            asc_in = str(out / "design.asc")
            bin_out = str(out / "design.bin")
            cmd = ["icepack", asc_in, bin_out]
            output_file = bin_out
        elif device.family == "ecp5":
            config_in = str(out / "design.config")
            bit_out = str(out / "design.bit")
            cmd = ["ecppack", "--compress", config_in, bit_out]
            output_file = bit_out
        else:
            return StageResult(
                name="Bitstream",
                passed=False,
                duration_s=0,
                message=f"Unsupported device family: {device.family}",
            )

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
        )

        if result.returncode != 0:
            return StageResult(
                name="Bitstream",
                passed=False,
                duration_s=time.time() - t0,
                message="bitstream generation failed",
                log=result.stderr,
            )

        size_kb = Path(output_file).stat().st_size / 1024
        return StageResult(
            name="Bitstream",
            passed=True,
            duration_s=time.time() - t0,
            output_files=[output_file],
            message=f"{size_kb:.1f} KB",
        )
    except Exception as e:
        return StageResult(
            name="Bitstream",
            passed=False,
            duration_s=time.time() - t0,
            message=str(e),
            log=str(e),
        )
