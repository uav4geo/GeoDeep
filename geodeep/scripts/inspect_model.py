import argparse
import sys
try:
    from geodeep.inference import create_session
    from geodeep import __version__
    from geodeep.models import get_model_file
except ImportError:
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from geodeep.inference import create_session
    from geodeep import __version__
    from geodeep.models import get_model_file

def main():
    parser = argparse.ArgumentParser(prog="geodeep-inspect", description="Inspect GeoDeep ONNX models")
    parser.add_argument(
        "model", 
        type=str,
        nargs="?",
        help="Model type or path to onnx model"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        required=False,
        help="Print version and exit"
    )
    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)
    
    if not args.model:
        parser.print_help(sys.stderr)
        exit(1)
    
    session, config = create_session(get_model_file(args.model))
    for k in config:
        print(f"{k}: {config[k]}")
    
if __name__ == "__main__":
    main()
