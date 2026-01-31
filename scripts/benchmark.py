#!/usr/bin/env python3
"""
Benchmarking script for Bar Chart Extractor.

Evaluates extraction accuracy against ground truth annotations.

Usage:
    python scripts/benchmark.py [--limit N] [--output results.csv] [--verbose]

Example:
    python scripts/benchmark.py --limit 10 --verbose
    python scripts/benchmark.py --output benchmark_results.csv
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.bar_extractor import BarChartExtractor


@dataclass
class BenchmarkResult:
    """Result for a single image benchmark."""
    image_path: str
    success: bool = False
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0

    # Ground truth
    gt_bar_count: int = 0
    gt_values: List[float] = field(default_factory=list)
    gt_categories: List[str] = field(default_factory=list)
    gt_title: Optional[str] = None

    # Predicted
    pred_bar_count: int = 0
    pred_values: List[float] = field(default_factory=list)
    pred_categories: List[str] = field(default_factory=list)
    pred_title: Optional[str] = None

    # Metrics
    bar_count_correct: bool = False
    value_mae: Optional[float] = None
    value_mape: Optional[float] = None
    category_accuracy: Optional[float] = None
    title_match: bool = False


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark run."""
    total_images: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0

    bar_count_accuracy: float = 0.0
    avg_value_mae: float = 0.0
    avg_value_mape: float = 0.0
    avg_category_accuracy: float = 0.0
    title_match_rate: float = 0.0

    avg_processing_time_ms: float = 0.0
    total_processing_time_s: float = 0.0


def load_annotations(annotations_path: Path) -> List[Dict[str, Any]]:
    """Load ground truth annotations from JSON file."""
    with open(annotations_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_mae(gt_values: List[float], pred_values: List[float]) -> Optional[float]:
    """Calculate Mean Absolute Error between ground truth and predicted values."""
    if not gt_values or not pred_values:
        return None

    # Match by position (assuming same order)
    n = min(len(gt_values), len(pred_values))
    if n == 0:
        return None

    errors = [abs(gt_values[i] - pred_values[i]) for i in range(n)]
    return sum(errors) / n


def calculate_mape(gt_values: List[float], pred_values: List[float]) -> Optional[float]:
    """Calculate Mean Absolute Percentage Error."""
    if not gt_values or not pred_values:
        return None

    n = min(len(gt_values), len(pred_values))
    if n == 0:
        return None

    errors = []
    for i in range(n):
        if gt_values[i] != 0:
            error = abs(gt_values[i] - pred_values[i]) / abs(gt_values[i]) * 100
            errors.append(error)

    return sum(errors) / len(errors) if errors else None


def calculate_category_accuracy(
    gt_categories: List[str],
    pred_categories: List[str]
) -> Optional[float]:
    """Calculate category matching accuracy."""
    if not gt_categories or not pred_categories:
        return None

    n = min(len(gt_categories), len(pred_categories))
    if n == 0:
        return None

    # Normalize and compare
    matches = 0
    for i in range(n):
        gt_cat = gt_categories[i].lower().strip()
        pred_cat = pred_categories[i].lower().strip()

        # Exact match or substring match
        if gt_cat == pred_cat or gt_cat in pred_cat or pred_cat in gt_cat:
            matches += 1

    return matches / n * 100


def title_matches(gt_title: Optional[str], pred_title: Optional[str]) -> bool:
    """Check if titles match (case-insensitive, partial match allowed)."""
    if not gt_title or not pred_title:
        return False

    gt_lower = gt_title.lower().strip()
    pred_lower = pred_title.lower().strip()

    # Exact match or significant overlap
    if gt_lower == pred_lower:
        return True

    # Check if ground truth words appear in prediction
    gt_words = set(gt_lower.split())
    pred_words = set(pred_lower.split())

    if len(gt_words) == 0:
        return False

    overlap = len(gt_words & pred_words) / len(gt_words)
    return overlap >= 0.5


def benchmark_single_image(
    extractor: BarChartExtractor,
    image_path: Path,
    annotation: Dict[str, Any],
    verbose: bool = False
) -> BenchmarkResult:
    """Benchmark extraction on a single image."""
    result = BenchmarkResult(image_path=str(image_path))

    # Extract ground truth
    metadata = annotation.get('metadata', {})
    result.gt_bar_count = len(metadata.get('values', []))
    result.gt_values = metadata.get('values', [])
    result.gt_categories = metadata.get('categories', [])
    result.gt_title = metadata.get('title')

    # Run extraction
    start_time = time.perf_counter()
    try:
        output = extractor.extract(str(image_path))
        result.success = True

        # Extract predictions
        data = output.get('data', [])
        result.pred_bar_count = len(data)
        result.pred_values = [item.get('value', 0) for item in data]
        result.pred_categories = [item.get('category', '') for item in data]
        result.pred_title = output.get('title')

    except Exception as e:
        result.success = False
        result.error_message = str(e)
        if verbose:
            print(f"  ERROR: {e}")

    end_time = time.perf_counter()
    result.processing_time_ms = (end_time - start_time) * 1000

    # Calculate metrics if successful
    if result.success:
        result.bar_count_correct = (result.gt_bar_count == result.pred_bar_count)
        result.value_mae = calculate_mae(result.gt_values, result.pred_values)
        result.value_mape = calculate_mape(result.gt_values, result.pred_values)
        result.category_accuracy = calculate_category_accuracy(
            result.gt_categories, result.pred_categories
        )
        result.title_match = title_matches(result.gt_title, result.pred_title)

    return result


def run_benchmark(
    annotations_path: Path,
    data_dir: Path,
    limit: Optional[int] = None,
    verbose: bool = False
) -> Tuple[List[BenchmarkResult], BenchmarkSummary]:
    """Run benchmark on all annotated images."""
    # Load annotations
    annotations = load_annotations(annotations_path)
    if limit:
        annotations = annotations[:limit]

    print(f"Loaded {len(annotations)} annotations")
    print(f"Data directory: {data_dir}")
    print("-" * 60)

    # Initialize extractor
    print("Initializing BarChartExtractor...")
    extractor = BarChartExtractor()
    print("Extractor ready.")
    print("-" * 60)

    results: List[BenchmarkResult] = []

    for i, annotation in enumerate(annotations):
        image_rel_path = annotation.get('image', '')
        image_path = data_dir / image_rel_path

        if not image_path.exists():
            if verbose:
                print(f"[{i+1}/{len(annotations)}] SKIP: {image_rel_path} (not found)")
            continue

        if verbose:
            print(f"[{i+1}/{len(annotations)}] Processing: {image_rel_path}")

        result = benchmark_single_image(extractor, image_path, annotation, verbose)
        results.append(result)

        if verbose:
            status = "OK" if result.success else "FAIL"
            bars = f"bars: {result.pred_bar_count}/{result.gt_bar_count}"
            time_str = f"{result.processing_time_ms:.0f}ms"
            print(f"  [{status}] {bars}, {time_str}")

    # Calculate summary
    summary = calculate_summary(results)

    return results, summary


def calculate_summary(results: List[BenchmarkResult]) -> BenchmarkSummary:
    """Calculate summary statistics from results."""
    summary = BenchmarkSummary()
    summary.total_images = len(results)

    if not results:
        return summary

    successful = [r for r in results if r.success]
    summary.successful_extractions = len(successful)
    summary.failed_extractions = summary.total_images - summary.successful_extractions

    if successful:
        # Bar count accuracy
        bar_correct = sum(1 for r in successful if r.bar_count_correct)
        summary.bar_count_accuracy = bar_correct / len(successful) * 100

        # Value MAE
        maes = [r.value_mae for r in successful if r.value_mae is not None]
        summary.avg_value_mae = sum(maes) / len(maes) if maes else 0.0

        # Value MAPE
        mapes = [r.value_mape for r in successful if r.value_mape is not None]
        summary.avg_value_mape = sum(mapes) / len(mapes) if mapes else 0.0

        # Category accuracy
        cat_accs = [r.category_accuracy for r in successful if r.category_accuracy is not None]
        summary.avg_category_accuracy = sum(cat_accs) / len(cat_accs) if cat_accs else 0.0

        # Title match rate
        title_matches = sum(1 for r in successful if r.title_match)
        summary.title_match_rate = title_matches / len(successful) * 100

        # Processing time
        times = [r.processing_time_ms for r in successful]
        summary.avg_processing_time_ms = sum(times) / len(times)
        summary.total_processing_time_s = sum(times) / 1000

    return summary


def print_summary(summary: BenchmarkSummary) -> None:
    """Print benchmark summary to console."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\nImages processed: {summary.total_images}")
    print(f"  Successful: {summary.successful_extractions}")
    print(f"  Failed: {summary.failed_extractions}")

    if summary.successful_extractions > 0:
        print(f"\nAccuracy Metrics:")
        print(f"  Bar count accuracy: {summary.bar_count_accuracy:.1f}%")
        print(f"  Value MAE: {summary.avg_value_mae:.2f}")
        print(f"  Value MAPE: {summary.avg_value_mape:.1f}%")
        print(f"  Category accuracy: {summary.avg_category_accuracy:.1f}%")
        print(f"  Title match rate: {summary.title_match_rate:.1f}%")

        print(f"\nPerformance:")
        print(f"  Avg processing time: {summary.avg_processing_time_ms:.0f}ms")
        print(f"  Total time: {summary.total_processing_time_s:.1f}s")

    print("=" * 60)


def export_results_csv(results: List[BenchmarkResult], output_path: Path) -> None:
    """Export detailed results to CSV file."""
    fieldnames = [
        'image_path', 'success', 'error_message', 'processing_time_ms',
        'gt_bar_count', 'pred_bar_count', 'bar_count_correct',
        'value_mae', 'value_mape', 'category_accuracy', 'title_match',
        'gt_title', 'pred_title'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                'image_path': r.image_path,
                'success': r.success,
                'error_message': r.error_message or '',
                'processing_time_ms': f"{r.processing_time_ms:.1f}",
                'gt_bar_count': r.gt_bar_count,
                'pred_bar_count': r.pred_bar_count,
                'bar_count_correct': r.bar_count_correct,
                'value_mae': f"{r.value_mae:.2f}" if r.value_mae else '',
                'value_mape': f"{r.value_mape:.1f}" if r.value_mape else '',
                'category_accuracy': f"{r.category_accuracy:.1f}" if r.category_accuracy else '',
                'title_match': r.title_match,
                'gt_title': r.gt_title or '',
                'pred_title': r.pred_title or ''
            })

    print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Bar Chart Extractor against ground truth'
    )
    parser.add_argument(
        '--annotations',
        type=Path,
        default=Path('data/annotations/bar_charts.json'),
        help='Path to annotations JSON file'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('.'),
        help='Base directory for image paths'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of images to process'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output CSV file for detailed results'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.annotations.exists():
        print(f"Error: Annotations file not found: {args.annotations}")
        sys.exit(1)

    # Run benchmark
    results, summary = run_benchmark(
        annotations_path=args.annotations,
        data_dir=args.data_dir,
        limit=args.limit,
        verbose=args.verbose
    )

    # Print summary
    print_summary(summary)

    # Export results if requested
    if args.output:
        export_results_csv(results, args.output)


if __name__ == '__main__':
    main()
