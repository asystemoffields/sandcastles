"""
Tests for CLI argument parsing (w2s.__main__).

These tests only exercise the argument parser, they do not run full compilation
(that would require a model file on disk).
"""

import pytest

from w2s.__main__ import build_parser


class TestCLIParser:
    def test_compile_command_accepted(self):
        parser = build_parser()
        args = parser.parse_args(["compile", "model.onnx"])
        assert args.command == "compile"
        assert args.model == "model.onnx"

    def test_compile_with_all_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "compile", "model.onnx",
            "--output", "/tmp/out",
            "--mode", "sequential",
            "--bits", "4",
            "--name", "my_net",
        ])
        assert args.output == "/tmp/out"
        assert args.mode == "sequential"
        assert args.bits == 4
        assert args.name == "my_net"

    def test_compile_default_options(self):
        parser = build_parser()
        args = parser.parse_args(["compile", "model.onnx"])
        assert args.output == "./output"
        assert args.mode == "auto"
        assert args.bits == 8
        assert args.name is None

    def test_estimate_command_accepted(self):
        parser = build_parser()
        args = parser.parse_args(["estimate", "model.onnx"])
        assert args.command == "estimate"
        assert args.model == "model.onnx"

    def test_estimate_default_mode_is_both(self):
        parser = build_parser()
        args = parser.parse_args(["estimate", "model.onnx"])
        assert args.mode == "both"

    def test_info_command_accepted(self):
        parser = build_parser()
        args = parser.parse_args(["info", "model.onnx"])
        assert args.command == "info"
        assert args.model == "model.onnx"

    def test_no_command_gives_none(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_invalid_bits_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compile", "model.onnx", "--bits", "3"])

    def test_invalid_mode_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compile", "model.onnx", "--mode", "turbo"])

    def test_compile_accepts_short_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "compile", "model.onnx",
            "-o", "/tmp/out",
            "-m", "combinational",
            "-b", "16",
            "-n", "net",
        ])
        assert args.output == "/tmp/out"
        assert args.mode == "combinational"
        assert args.bits == 16
        assert args.name == "net"

    def test_compile_with_fpga_target(self):
        parser = build_parser()
        args = parser.parse_args([
            "compile", "model.onnx",
            "--target", "fpga",
            "--device", "ecp5-25k",
        ])
        assert args.command == "compile"
        assert args.target == "fpga"
        assert args.device == "ecp5-25k"

    def test_compile_with_bits_map(self):
        parser = build_parser()
        args = parser.parse_args([
            "compile", "model.onnx",
            "--bits-map", "hidden=4,output=16",
        ])
        assert args.command == "compile"
        assert args.bits_map == "hidden=4,output=16"

    # --- autofit ---

    def test_autofit_command_accepted(self):
        parser = build_parser()
        args = parser.parse_args(["autofit", "model.onnx"])
        assert args.command == "autofit"
        assert args.model == "model.onnx"

    def test_autofit_with_device(self):
        parser = build_parser()
        args = parser.parse_args([
            "autofit", "model.onnx",
            "--device", "ecp5-25k",
        ])
        assert args.command == "autofit"
        assert args.device == "ecp5-25k"

    # --- build ---

    def test_build_command_accepted(self):
        parser = build_parser()
        args = parser.parse_args(["build", "model.onnx"])
        assert args.command == "build"
        assert args.model == "model.onnx"

    def test_build_with_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "build", "model.onnx",
            "--device", "ice40up5k",
            "--no-simulate",
            "--bits", "4",
        ])
        assert args.command == "build"
        assert args.device == "ice40up5k"
        assert args.no_simulate is True
        assert args.bits == 4

    # --- testbench ---

    def test_testbench_command_accepted(self):
        parser = build_parser()
        args = parser.parse_args(["testbench", "model.onnx"])
        assert args.command == "testbench"
        assert args.model == "model.onnx"

    def test_testbench_with_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "testbench", "model.onnx",
            "--vectors", "8",
            "--vcd",
            "--tolerance", "2",
        ])
        assert args.command == "testbench"
        assert args.vectors == 8
        assert args.vcd is True
        assert args.tolerance == 2
