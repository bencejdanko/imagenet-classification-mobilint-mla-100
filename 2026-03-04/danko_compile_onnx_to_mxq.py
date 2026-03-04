import argparse
import os
from qubee.compiler.compiler import Compiler
from qubee.configs import CompileConfig


def main():
    parser = argparse.ArgumentParser(description="Compile ONNX model with Qubee compiler.")
    parser.add_argument("--model", default="test_model/imagenetsub20resnet10-5.onnx", help="Path to input ONNX model")
    parser.add_argument("--output", default="compiled_model.mxq", help="Path to save compiled MXQ model")
    parser.add_argument("--calib", default="calib_npy", help="Path to calibration data directory or txt list")
    args = parser.parse_args()

    use_random = not os.path.exists(args.calib)
    if use_random:
        print(f"Warning: Calibration data {args.calib} not found. Using random calibration.")
    else:
        print(f"Using calibration data from {args.calib}")

    c_config = CompileConfig.default_config(
        use_random_calib=use_random,
        calib_data_path=args.calib if not use_random else None
    )
    c_config.save_path = args.output
    c_config.quantization.calibration.quantization_mode = 2 # maxPercentile
    c_config.quantization.calibration.percentile = 0.999
    c_config.quantization.calibration.quantization_method = 1 # WChALayer
    c_config.quantization.calibration.quantization_output = 0 # Layer

    compiler = Compiler(
        model=args.model,
        backend="onnx",
        target_device="aries2"
    )

    compiler.compile(
        save_path=args.output,
        use_random_calib=use_random,
        calib_data_path=args.calib if not use_random else None,
        compile_config=c_config,
        compile_nde=False
    )

    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()