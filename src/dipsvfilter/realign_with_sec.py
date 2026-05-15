#!/usr/bin/env python3
"""
Extract reads overlapping a constructed-haplotype cluster and realign them.

The input FASTA is expected to be produced by ``bin/construct_haplotypes.py``,
whose filenames encode the source cluster as ``chrom_start-end.fasta``. Reads
overlapping that original-reference interval are extracted from the input BAM,
deduplicated by read name, and aligned back to the constructed contigs with
minimap2 while retaining multiple secondary alignments.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pysam


def parse_cluster_from_fasta(fasta_path: str) -> Tuple[str, int, int]:
    """Parse ``chrom_start-end`` cluster coordinates from a FASTA basename."""

    stem = Path(fasta_path).stem
    try:
        chrom, interval = stem.rsplit("_", 1)
        start_text, end_text = interval.split("-", 1)
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise ValueError(
            "Could not parse cluster coordinates from FASTA basename. "
            "Expected a name like chrom_start-end.fasta, for example "
            "chr1_10000-11000.fasta."
        ) from exc

    if start < 0 or end <= start:
        raise ValueError(f"Invalid cluster interval parsed from FASTA: {stem}")

    return chrom, start, end


def cluster_name_from_fasta(fasta_path: str) -> str:
    """Return the output basename derived from the input FASTA."""

    return Path(fasta_path).stem


def count_fasta_records(fasta_path: str) -> int:
    """Count records in a plain-text FASTA file."""

    count = 0
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1

    if count == 0:
        raise ValueError(f"No contigs found in FASTA: {fasta_path}")

    return count


def sequence_priority(record: pysam.AlignedSegment) -> Tuple[int, int, int]:
    """Rank candidate records for choosing one sequence per read."""

    if not record.is_secondary and not record.is_supplementary:
        record_type_priority = 2
    elif record.is_supplementary:
        record_type_priority = 1
    else:
        record_type_priority = 0

    sequence_length = len(record.query_sequence) if record.query_sequence else 0
    return (record_type_priority, sequence_length, record.mapping_quality)


def collect_region_read_sequences(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
) -> Dict[str, str]:
    """Collect one sequence per read with any alignment overlapping a region."""

    read_sequences: Dict[str, str] = {}
    read_priorities: Dict[str, Tuple[int, int, int]] = {}

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        try:
            records = bam.fetch(chrom, start, end)
        except ValueError as exc:
            raise ValueError(f"Region {chrom}:{start}-{end} is not fetchable from BAM") from exc

        for record in records:
            if record.is_unmapped or not record.query_sequence:
                continue

            priority = sequence_priority(record)
            if priority > read_priorities.get(record.query_name, (-1, -1, -1)):
                read_sequences[record.query_name] = record.query_sequence.upper()
                read_priorities[record.query_name] = priority

    return read_sequences


def write_reads_fasta(read_sequences: Dict[str, str], output_path: str, line_width: int = 80) -> None:
    """Write deduplicated read sequences to FASTA."""

    with open(output_path, "w", encoding="utf-8") as handle:
        for read_name in sorted(read_sequences):
            sequence = read_sequences[read_name]
            handle.write(f">{read_name}\n")
            for i in range(0, len(sequence), line_width):
                handle.write(sequence[i : i + line_width] + "\n")


def run_minimap2(
    minimap2: str,
    contig_fasta: str,
    reads_fasta: str,
    sam_path: str,
    num_contigs: int,
    threads: int,
    preset: str,
    soft_clip_supplementary: bool,
) -> None:
    """Run minimap2 and write SAM output to ``sam_path``."""

    command = [
        minimap2,
        "-a",
        "-x",
        preset,
        "-N",
        str(num_contigs),
        "-f",
        "0.05",
        "-t",
        str(threads),
    ]
    if soft_clip_supplementary:
        command.append("-Y")
    command.extend([contig_fasta, reads_fasta])

    with open(sam_path, "w", encoding="utf-8") as sam_handle:
        subprocess.run(command, stdout=sam_handle, check=True)


def sort_and_index_alignment(sam_path: str, bam_path: str, threads: int, index: bool) -> None:
    """Sort SAM into BAM and optionally create a BAM index."""

    pysam.sort("-@", str(threads), "-o", bam_path, sam_path)
    if index:
        pysam.index(bam_path)


def realign_cluster(args: argparse.Namespace) -> str:
    """Extract region-overlapping reads and realign them to constructed contigs."""

    if args.threads <= 0:
        raise ValueError("--threads must be a positive integer")

    chrom, start, end = parse_cluster_from_fasta(args.fasta)
    num_contigs = count_fasta_records(args.fasta)
    cluster_name = cluster_name_from_fasta(args.fasta)

    os.makedirs(args.out_dir, exist_ok=True)
    output_bam = os.path.join(args.out_dir, f"{cluster_name}.bam")
    reads_fasta = os.path.join(args.out_dir, f"{cluster_name}.reads.fasta")

    read_sequences = collect_region_read_sequences(args.bam, chrom, start, end)
    if not read_sequences:
        raise ValueError(f"No reads with sequences found in {chrom}:{start}-{end}")

    write_reads_fasta(read_sequences, reads_fasta)

    with tempfile.TemporaryDirectory(prefix=f"{cluster_name}.", dir=args.tmp_dir or args.out_dir) as tmp_dir:
        sam_path = os.path.join(tmp_dir, f"{cluster_name}.sam")
        run_minimap2(
            args.minimap2,
            args.fasta,
            reads_fasta,
            sam_path,
            num_contigs,
            args.threads,
            args.preset,
            args.soft_clip_supplementary,
        )
        sort_and_index_alignment(sam_path, output_bam, args.threads, not args.no_index)

    if not args.keep_reads:
        os.remove(reads_fasta)

    print(f"Cluster: {chrom}:{start}-{end}")
    print(f"Constructed contigs: {num_contigs}")
    print(f"Extracted unique reads: {len(read_sequences)}")
    print(f"Wrote BAM: {output_bam}")
    if not args.no_index:
        print(f"Wrote BAM index: {output_bam}.bai")

    return output_bam


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract one sequence per read from an original BAM over the cluster "
            "encoded in a constructed-contig FASTA name, then realign those reads "
            "to the constructed contigs with minimap2 secondary alignments enabled."
        )
    )
    parser.add_argument("--fasta", required=True, help="Constructed contig FASTA")
    parser.add_argument("--bam", required=True, help="Original read alignment BAM")
    parser.add_argument("--out-dir", default=".", help="Output directory (default: current directory)")
    parser.add_argument("--threads", "-t", type=int, default=1, help="Threads for minimap2 and sorting")
    parser.add_argument("--minimap2", default="minimap2", help="Path to minimap2 executable")
    parser.add_argument("--preset", "-x", default="map-hifi", help="minimap2 -x preset (default: map-hifi)")
    parser.add_argument(
        "--soft-clip-supplementary",
        action="store_true",
        help="Add minimap2 -Y to soft-clip supplementary alignments",
    )
    parser.add_argument(
        "--keep-reads",
        action="store_true",
        help="Keep the intermediate deduplicated reads FASTA",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Do not create a BAM index",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory for temporary SAM files (default: output directory)",
    )
    return parser


if __name__ == "__main__":
    realign_cluster(build_arg_parser().parse_args())
