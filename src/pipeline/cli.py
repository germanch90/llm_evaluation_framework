"""
Command-line interface for exercising the RAG pipeline.

Usage:
    python -m src.pipeline.cli ingest --input data/kg-rag-dataset/legal
    python -m src.pipeline.cli query --question "What is the warranty period?"
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG pipeline CLI utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF documents")
    ingest_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing PDF files",
    )

    query_parser = subparsers.add_parser("query", help="Run a query against the pipeline")
    query_parser.add_argument(
        "--question",
        "-q",
        required=True,
        help="User question to answer",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )

    return parser


def cmd_ingest(pipeline: RAGPipeline, input_dir: Path) -> None:
    stats = pipeline.ingest_directory(input_dir)
    logger.info("Ingestion stats: %s", stats)


def cmd_query(pipeline: RAGPipeline, question: str, top_k: int) -> None:
    result = asyncio.run(pipeline.query_async(question=question, top_k=top_k))
    logger.info("Answer: %s", result.answer)
    print("\nAnswer:\n--------\n", result.answer)
    if result.contexts:
        print("\nSources:")
        for chunk in result.contexts:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page")
            page_info = f" (page {page})" if page else ""
            print(f"- {chunk.id}: {source}{page_info} | score={chunk.score:.3f}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = RAGPipeline()

    if args.command == "ingest":
        cmd_ingest(pipeline, args.input)
    elif args.command == "query":
        cmd_query(pipeline, args.question, args.top_k)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
