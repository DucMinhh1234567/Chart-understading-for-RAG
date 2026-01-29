"""
Chart Understanding - Bar Chart Data Extraction

Extract structured data from bar chart images using computer vision and OCR.

Usage:
    python main.py <image_path> [--output output.json] [--ocr easyocr|tesseract]

Examples:
    python main.py data/raw/bar_charts/chart_0001.png
    python main.py chart.png -o result.json
    python main.py chart.png --ocr tesseract
"""
import argparse
import json
import sys
from pathlib import Path

from src.extraction.bar_extractor import BarChartExtractor


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Extract data from bar chart images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py chart.png                    # Print JSON to stdout
  python main.py chart.png -o result.json     # Save to file
  python main.py chart.png --ocr tesseract    # Use Tesseract OCR
        """
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to bar chart image (PNG, JPG, JPEG, BMP)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: print to stdout)"
    )
    parser.add_argument(
        "--ocr",
        type=str,
        choices=["easyocr", "tesseract"],
        default="easyocr",
        help="OCR engine to use (default: easyocr)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        return 1
    
    # Validate image format
    supported_formats = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    if image_path.suffix.lower() not in supported_formats:
        print(
            f"Error: Unsupported image format: {image_path.suffix}\n"
            f"Supported formats: {', '.join(supported_formats)}",
            file=sys.stderr
        )
        return 1
    
    # Extract data from chart
    try:
        extractor = BarChartExtractor()
        result = extractor.extract(str(image_path), ocr_method=args.ocr)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Failed to extract data: {e}", file=sys.stderr)
        return 1
    
    # Format output as JSON
    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    
    # Write to file or stdout
    if args.output:
        output_path = Path(args.output)
        try:
            output_path.write_text(output_json, encoding="utf-8")
            print(f"Result saved to: {args.output}")
        except IOError as e:
            print(f"Error: Failed to write output file: {e}", file=sys.stderr)
            return 1
    else:
        print(output_json)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
