#!/usr/bin/env python3
"""
Rank contig pairs by agreement between contig and read k-mer profiles.

The script evaluates all unordered contig pairs, including self-pairs, from a
candidate-contig FASTA. Reads are taken from a BAM in which the reads have been
aligned to those contigs. For each pair, reads with evidence for both contigs
are assigned to the better-supported contig, then canonical k-mer frequency
vectors are compared with cosine distance.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pysam


DNA_ALPHABET = "ACGT"
REVCOMP_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")
ReadContigKey = Tuple[str, str]


@dataclass
class ReadContigEvidence:
    """Scoring evidence for one read aligned to one contig."""

    contig: str
    query_start: Optional[int] = None
    query_end: Optional[int] = None
    max_mapq: int = -1
    aligned_query_bases: int = 0
    nm_sum: Optional[int] = None
    kmer_vector: Optional[np.ndarray] = None

    def add_record(self, record: pysam.AlignedSegment, query_span: Tuple[int, int]) -> None:
        """Add alignment-derived metadata to this evidence object."""

        start, end = query_span
        self.query_start = start if self.query_start is None else min(self.query_start, start)
        self.query_end = end if self.query_end is None else max(self.query_end, end)
        self.max_mapq = max(self.max_mapq, record.mapping_quality)
        self.aligned_query_bases += record.query_alignment_length or max(0, end - start)

        if record.has_tag("NM"):
            nm = int(record.get_tag("NM"))
            self.nm_sum = nm if self.nm_sum is None else self.nm_sum + nm

    def finalize(self, read_sequence: Optional[str], kmer_index: Dict[str, int], k: int) -> None:
        """Build the k-mer vector from the contig-supported read span."""

        if self.query_start is None or self.query_end is None or not read_sequence:
            self.kmer_vector = np.zeros(len(kmer_index), dtype=np.float64)
            return

        start = max(0, self.query_start)
        end = min(len(read_sequence), self.query_end)
        self.kmer_vector = count_kmers(read_sequence[start:end], kmer_index, k)


@dataclass(frozen=True)
class PairScore:
    """Scoring result for one candidate contig pair."""

    contig1: str
    contig2: str
    cosine_distance: float
    reads_used: int
    read_kmers: int
    contig_kmers: int
    output_bams: str = ""


def read_fasta(fasta_path: str) -> Dict[str, str]:
    """Read a FASTA file into an ordered dictionary-like mapping."""

    sequences: Dict[str, List[str]] = {}
    current_name: Optional[str] = None

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_name = line[1:].split()[0]
                if current_name in sequences:
                    raise ValueError(f"duplicate FASTA contig name: {current_name}")
                sequences[current_name] = []
            elif current_name is None:
                raise ValueError("FASTA sequence encountered before first header")
            else:
                sequences[current_name].append(line.upper())

    if not sequences:
        raise ValueError(f"no contigs found in FASTA: {fasta_path}")

    return {name: "".join(parts) for name, parts in sequences.items()}


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""

    return seq.translate(REVCOMP_TABLE)[::-1].upper()


def canonical_kmer(kmer: str) -> str:
    """Collapse a k-mer and its reverse complement to one canonical label."""

    kmer = kmer.upper()
    return min(kmer, reverse_complement(kmer))


def build_canonical_kmer_index(k: int) -> Dict[str, int]:
    """Create a stable index for all canonical A/C/G/T k-mers of length k."""

    if k <= 0:
        raise ValueError("k must be a positive integer")

    labels = {
        canonical_kmer("".join(chars))
        for chars in itertools.product(DNA_ALPHABET, repeat=k)
    }
    return {kmer: index for index, kmer in enumerate(sorted(labels))}


def count_kmers(seq: str, kmer_index: Dict[str, int], k: int) -> np.ndarray:
    """Count canonical k-mers in a sequence, skipping non-ACGT k-mers."""

    vector = np.zeros(len(kmer_index), dtype=np.float64)
    seq = seq.upper()
    if len(seq) < k:
        return vector

    valid_bases = set(DNA_ALPHABET)
    for i in range(0, len(seq) - k + 1):
        kmer = seq[i : i + k]
        if any(base not in valid_bases for base in kmer):
            continue
        vector[kmer_index[canonical_kmer(kmer)]] += 1.0

    return vector


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Convert a count vector to a frequency vector."""

    total = vector.sum()
    if total == 0:
        return vector.copy()
    return vector / total


def cosine_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Return cosine distance, with NaN for empty vectors."""

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return math.nan
    similarity = float(np.dot(vector1, vector2) / (norm1 * norm2))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def collect_read_sequences(bam_path: str) -> Dict[str, str]:
    """
    Collect one primary-alignment SEQ-field sequence for each read in the BAM.

    This script expects the input BAM to be the direct minimap2 realignment BAM,
    not a regional subset from samtools view. Under that assumption, mapped reads
    should have a primary alignment carrying the read sequence. The stored
    sequence is not reoriented because canonical k-mer counting makes strand
    orientation irrelevant, and keeping SEQ/CIGAR coordinates in the same
    orientation keeps slicing simple.
    """

    read_sequences: Dict[str, str] = {}

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for record in bam.fetch(until_eof=True):
            if record.is_unmapped or record.is_secondary or record.is_supplementary:
                continue
            sequence = record.query_sequence
            if not sequence:
                continue
            if record.query_name in read_sequences:
                continue

            read_sequences[record.query_name] = sequence.upper()

    return read_sequences


def aligned_query_span(record: pysam.AlignedSegment) -> Optional[Tuple[int, int]]:
    """
    Return the query-sequence span that is anchored to the reference.

    The span excludes leading/trailing soft or hard clipped sequence for this
    alignment. Hard clips are still counted as offsets, allowing supplementary
    records such as 900H100M to map to the expected coordinates when a full read
    sequence has been recovered from another record.
    """

    if record.cigartuples is None:
        return None

    query_pos = 0
    starts: List[int] = []
    ends: List[int] = []

    for op, length in record.cigartuples:
        consumes_query = op in (0, 1, 4, 5, 7, 8)
        consumes_ref = op in (0, 2, 3, 7, 8)

        if consumes_query and consumes_ref:
            starts.append(query_pos)
            ends.append(query_pos + length)

        if consumes_query:
            query_pos += length

    if not starts:
        return None

    return min(starts), max(ends)


def collect_read_evidence(
    bam_path: str,
    fasta_contigs: Sequence[str],
    read_sequences: Dict[str, str],
    kmer_index: Dict[str, int],
    k: int,
) -> Tuple[
    Dict[str, Dict[str, ReadContigEvidence]],
    Dict[ReadContigKey, List[pysam.AlignedSegment]],
    pysam.AlignmentHeader,
]:
    """Collect per-read, per-contig alignment evidence from the BAM."""

    fasta_contig_set = set(fasta_contigs)
    read_evidence: Dict[str, Dict[str, ReadContigEvidence]] = {}
    records_by_read_contig: Dict[ReadContigKey, List[pysam.AlignedSegment]] = {}
    missing_primary_sequence_records = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        header = bam.header.copy()
        missing_from_fasta = sorted(set(bam.references) - fasta_contig_set)
        if missing_from_fasta:
            preview = ", ".join(missing_from_fasta[:5])
            print(
                f"[WARNING] BAM contains references not present in FASTA; "
                f"they will be ignored: {preview}"
            )
        missing_from_bam = sorted(fasta_contig_set - set(bam.references))
        if missing_from_bam:
            preview = ", ".join(missing_from_bam[:5])
            print(
                f"[WARNING] FASTA contains contigs not present in BAM; "
                f"they can score only as zero-read pairs: {preview}"
            )

        for record in bam.fetch(until_eof=True):
            if record.is_unmapped:
                continue
            if record.reference_name not in fasta_contig_set:
                continue

            query_span = aligned_query_span(record)
            if query_span is None:
                continue
            if record.query_name not in read_sequences:
                missing_primary_sequence_records += 1
                continue

            per_read = read_evidence.setdefault(record.query_name, {})
            evidence = per_read.setdefault(
                record.reference_name,
                ReadContigEvidence(record.reference_name),
            )
            evidence.add_record(record, query_span)
            records_by_read_contig.setdefault(
                (record.query_name, record.reference_name),
                [],
            ).append(record)

    if missing_primary_sequence_records:
        print(
            "[WARNING] Skipped "
            f"{missing_primary_sequence_records} alignment records because their reads "
            "did not have a primary-alignment sequence in this BAM."
        )

    for read_name, per_read in read_evidence.items():
        for evidence in per_read.values():
            evidence.finalize(read_sequences.get(read_name), kmer_index, k)

    return read_evidence, records_by_read_contig, header


def evidence_sort_key(evidence: ReadContigEvidence) -> Tuple[int, int, int, str]:
    """Return a deterministic key for choosing the best read-contig assignment."""

    nm_score = -evidence.nm_sum if evidence.nm_sum is not None else -10**12
    return (evidence.max_mapq, evidence.aligned_query_bases, nm_score, evidence.contig)


def choose_evidence_for_pair(
    per_read: Dict[str, ReadContigEvidence],
    contig1: str,
    contig2: str,
) -> Optional[ReadContigEvidence]:
    """Choose the retained read-contig evidence for one pair."""

    if contig1 == contig2:
        return per_read.get(contig1)

    candidates = [
        evidence
        for contig, evidence in per_read.items()
        if contig == contig1 or contig == contig2
    ]
    if not candidates:
        return None

    return max(candidates, key=evidence_sort_key)


def score_pair(
    contig1: str,
    contig2: str,
    contig_vectors: Dict[str, np.ndarray],
    read_evidence: Dict[str, Dict[str, ReadContigEvidence]],
) -> PairScore:
    """Score one contig pair by cosine distance between normalized k-mer vectors."""

    read_vector = np.zeros_like(next(iter(contig_vectors.values())), dtype=np.float64)
    reads_used = 0

    for per_read in read_evidence.values():
        evidence = choose_evidence_for_pair(per_read, contig1, contig2)
        if evidence is None or evidence.kmer_vector is None:
            continue
        if evidence.kmer_vector.sum() == 0:
            continue
        read_vector += evidence.kmer_vector
        reads_used += 1

    contig_vector = contig_vectors[contig1] + contig_vectors[contig2]
    distance = cosine_distance(normalize_vector(read_vector), normalize_vector(contig_vector))

    return PairScore(
        contig1=contig1,
        contig2=contig2,
        cosine_distance=distance,
        reads_used=reads_used,
        read_kmers=int(read_vector.sum()),
        contig_kmers=int(contig_vector.sum()),
    )


def score_all_pairs(
    contigs: Sequence[str],
    contig_vectors: Dict[str, np.ndarray],
    read_evidence: Dict[str, Dict[str, ReadContigEvidence]],
    threads: int = 1,
) -> List[PairScore]:
    """Score every unordered contig pair, including self-pairs."""

    pairs = list(itertools.combinations_with_replacement(contigs, 2))
    if threads == 1:
        scores = [
            score_pair(contig1, contig2, contig_vectors, read_evidence)
            for contig1, contig2 in pairs
        ]
    else:
        chunksize = max(1, len(pairs) // (threads * 4))
        with ProcessPoolExecutor(
            max_workers=threads,
            initializer=init_score_worker,
            initargs=(contig_vectors, read_evidence),
        ) as executor:
            scores = list(executor.map(score_pair_worker, pairs, chunksize=chunksize))

    scores = [score for score in scores if not math.isnan(score.cosine_distance)]
    return sorted(scores, key=lambda s: (s.cosine_distance, -s.reads_used, s.contig1, s.contig2))


_WORKER_CONTIG_VECTORS: Optional[Dict[str, np.ndarray]] = None
_WORKER_READ_EVIDENCE: Optional[Dict[str, Dict[str, ReadContigEvidence]]] = None


def init_score_worker(
    contig_vectors: Dict[str, np.ndarray],
    read_evidence: Dict[str, Dict[str, ReadContigEvidence]],
) -> None:
    """Initialize multiprocessing workers with read-only scoring data."""

    global _WORKER_CONTIG_VECTORS, _WORKER_READ_EVIDENCE
    _WORKER_CONTIG_VECTORS = contig_vectors
    _WORKER_READ_EVIDENCE = read_evidence


def score_pair_worker(pair: Tuple[str, str]) -> PairScore:
    """Score one contig pair inside a multiprocessing worker."""

    if _WORKER_CONTIG_VECTORS is None or _WORKER_READ_EVIDENCE is None:
        raise RuntimeError("Score worker was not initialized")

    contig1, contig2 = pair
    return score_pair(contig1, contig2, _WORKER_CONTIG_VECTORS, _WORKER_READ_EVIDENCE)


def cluster_name_from_fasta(fasta_path: str) -> str:
    """Return the cluster name encoded by the FASTA basename."""

    basename = Path(fasta_path).name
    for suffix in (".fasta", ".fa", ".fna"):
        if basename.endswith(suffix):
            return basename[: -len(suffix)]
    return Path(fasta_path).stem


def contig_gt_label(contig_name: str) -> str:
    """Extract a compact genotype-combination label from a FASTA contig name."""

    for field in contig_name.split("|"):
        if field.startswith("GT="):
            gt = field.removeprefix("GT=")
            return "".join(gt.split(":"))

    return sanitize_filename(contig_name)


def sanitize_filename(name: str) -> str:
    """Return a filesystem-friendly fallback label."""

    keep = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    return "".join(char if char in keep else "_" for char in name)


def write_pair_bams(
    score: PairScore,
    rank: int,
    fasta_path: str,
    out_dir: str,
    header: pysam.AlignmentHeader,
    read_evidence: Dict[str, Dict[str, ReadContigEvidence]],
    records_by_read_contig: Dict[ReadContigKey, List[pysam.AlignedSegment]],
    sort_index: bool,
) -> List[str]:
    """Write retained alignments for one ranked pair to per-contig BAM files."""

    cluster_name = cluster_name_from_fasta(fasta_path)
    rank_name = f"{rank:02d}"
    rank_dir = os.path.join(out_dir, cluster_name, rank_name)
    os.makedirs(rank_dir, exist_ok=True)

    pair_contigs = [score.contig1] if score.contig1 == score.contig2 else [score.contig1, score.contig2]
    bam_paths = {
        contig: os.path.join(rank_dir, f"{contig_gt_label(contig)}.bam")
        for contig in pair_contigs
    }
    temp_paths = {
        contig: path + ".tmp.bam" if sort_index else path
        for contig, path in bam_paths.items()
    }

    writers = {
        contig: pysam.AlignmentFile(temp_path, "wb", header=header)
        for contig, temp_path in temp_paths.items()
    }
    try:
        for read_name, per_read in read_evidence.items():
            evidence = choose_evidence_for_pair(per_read, score.contig1, score.contig2)
            if evidence is None:
                continue
            if evidence.contig not in writers:
                continue
            for record in records_by_read_contig.get((read_name, evidence.contig), []):
                writers[evidence.contig].write(record)
    finally:
        for writer in writers.values():
            writer.close()

    if sort_index:
        for contig, final_bam in bam_paths.items():
            temp_bam = temp_paths[contig]
            pysam.sort("-o", final_bam, temp_bam)
            os.remove(temp_bam)
            pysam.index(final_bam)

    return [bam_paths[contig] for contig in pair_contigs]


def write_summary_tsv(
    scores: Sequence[PairScore],
    summary_path: str,
) -> None:
    """Write the rank-to-contig-pair mapping and score summary."""

    fields = [
        "rank",
        "output_bams",
        "contig1",
        "contig2",
        "self_pair",
        "partition_mode",
        "cosine_distance",
        "reads_used",
        "read_kmers",
        "contig_kmers",
    ]

    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rank, score in enumerate(scores, start=1):
            self_pair = score.contig1 == score.contig2
            writer.writerow(
                {
                    "rank": rank,
                    "output_bams": score.output_bams,
                    "contig1": score.contig1,
                    "contig2": score.contig2,
                    "self_pair": str(self_pair).lower(),
                    "partition_mode": "single_contig" if self_pair else "best_evidence_assignment",
                    "cosine_distance": f"{score.cosine_distance:.8g}",
                    "reads_used": score.reads_used,
                    "read_kmers": score.read_kmers,
                    "contig_kmers": score.contig_kmers,
                }
            )


def run(args: argparse.Namespace) -> None:
    """Execute the contig-pair k-mer filtering workflow."""

    if args.top_n <= 0:
        raise ValueError("--top-n must be a positive integer")
    if args.threads <= 0:
        raise ValueError("--threads must be a positive integer")

    os.makedirs(args.out_dir, exist_ok=True)

    contig_sequences = read_fasta(args.fasta)
    contigs = list(contig_sequences.keys())
    kmer_index = build_canonical_kmer_index(args.k)

    contig_vectors = {
        contig: count_kmers(sequence, kmer_index, args.k)
        for contig, sequence in contig_sequences.items()
    }

    read_sequences = collect_read_sequences(args.bam)
    read_evidence, records_by_read_contig, header = collect_read_evidence(
        args.bam,
        contigs,
        read_sequences,
        kmer_index,
        args.k,
    )

    expected_pairs = len(contigs) * (len(contigs) + 1) // 2
    scores = score_all_pairs(contigs, contig_vectors, read_evidence, args.threads)
    top_scores = scores[: args.top_n]

    ranked_scores: List[PairScore] = []
    for rank, score in enumerate(top_scores, start=1):
        output_paths = write_pair_bams(
            score,
            rank,
            args.fasta,
            args.out_dir,
            header,
            read_evidence,
            records_by_read_contig,
            sort_index=not args.no_sort_index,
        )
        ranked_scores.append(
            PairScore(
                contig1=score.contig1,
                contig2=score.contig2,
                cosine_distance=score.cosine_distance,
                reads_used=score.reads_used,
                read_kmers=score.read_kmers,
                contig_kmers=score.contig_kmers,
                output_bams=";".join(
                    os.path.relpath(output_path, args.out_dir)
                    for output_path in output_paths
                ),
            )
        )

    cluster_name = cluster_name_from_fasta(args.fasta)
    cluster_dir = os.path.join(args.out_dir, cluster_name)
    os.makedirs(cluster_dir, exist_ok=True)
    summary_path = os.path.join(cluster_dir, "top_pairs.tsv")
    write_summary_tsv(ranked_scores, summary_path)

    print(f"Loaded {len(contigs)} contigs from FASTA")
    print(f"Recovered full read sequences for {len(read_sequences)} reads")
    print(f"Expected pairs: {expected_pairs}; scored pairs with usable k-mers: {len(scores)}")
    print(f"Scoring threads: {args.threads}")
    print(f"Wrote top {len(ranked_scores)} ranked pair directories to {cluster_dir}")
    print(f"Wrote summary TSV: {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Rank candidate contig pairs using canonical k-mer frequency cosine "
            "distance and write filtered BAMs for the top-ranked pairs."
        )
    )
    parser.add_argument("--bam", required=True, help="Input BAM aligned to candidate contigs")
    parser.add_argument("--fasta", required=True, help="Candidate contig FASTA")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--k", type=int, default=4, help="k-mer size (default: 4)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top pairs to write (default: 10)")
    parser.add_argument("--threads", type=int, default=1, help="Number of worker processes for pair scoring")
    parser.add_argument(
        "--no-sort-index",
        action="store_true",
        help="Write BAMs without coordinate sorting and indexing",
    )
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
