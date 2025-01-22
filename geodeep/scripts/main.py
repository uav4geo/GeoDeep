import argparse
import sys
try:
    from geodeep import detect, models, simple_progress
except ImportError:
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from geodeep import detect, models, simple_progress

def main():
    parser = argparse.ArgumentParser(prog="geodeep", description="AI object detection in geospatial rasters ")
    parser.add_argument(
        "geotiff", 
        type=str, 
        nargs="?",
        help="Path to GeoTIFF"
    )
    parser.add_argument(
        "model", 
        type=str, 
        nargs="?",
        help="Model type or path to onnx model"
    )

    parser.add_argument(
        "--output-type", "-t",
        type=str, 
        default="geojson",
        choices=["geojson", "bsc", "raw"], 
        required=False, 
        help='Type of output. One of: %(choices)s'
    )
    parser.add_argument(
        "--geojson-output", "-o", 
        type=str, 
        default="boxes.geojson",
        required=False, 
        help='GeoJSON output filename. Default: %(default)s'
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        required=False,
        help="Print list of available models and exit"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        required=False,
        help="Don't print progress"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        required=False,
        help="Print version and exit"
    )
    args = parser.parse_args()

    if args.list_models:
        print([k for k in models.list_models()])
        exit(0)
    
    if args.version:
        import warnings
        warnings.simplefilter("ignore", DeprecationWarning)

        import pkg_resources
        try:
            print(pkg_resources.get_distribution('geodeep').version)
        except:
            print("dev")
        exit(0)
    
    if not args.model or not args.geotiff:
        parser.print_help(sys.stderr)
        exit(1)
    
    output = detect(args.geotiff, args.model, 
                    output_type=args.output_type, 
                    progress_callback=simple_progress if not args.quiet else None)
    
    if args.output_type == "geojson":
        with open(args.geojson_output, "w") as f:
            f.write(output)
        print(f"Wrote {args.geojson_output}")
    else:
        print(output)

if __name__ == "__main__":
    main()