# credit: @ yichen

import argparse
import numpy as np
import pysam
import re
import os


def logify_numpy(
    a: np.ndarray,
) -> np.ndarray:
    """Logarithmic transformation used in MAMNET normalization."""

    return np.log(((a > 0) * a) + 1.0) - np.log((np.abs(a) * (a < 0)) + 1.0)


def trim_cigar(
    segment: pysam.AlignedSegment,
    region_start: int,
    region_end: int,
) -> list[tuple[int, int]]:
    """Return the portion of a CIGAR string overlapping a target reference region."""

    ref_pos = segment.reference_start

    trimmed_cigar = list()

    for op, length in segment.cigartuples:
        if (
            op
            in (
                0,
                2,
                3,
                7,
                8,
            )
        ):  # M, D, N, =, X: consumes reference (M, D, N, =, X) and possibly query (M, =, X)
            ref_end = ref_pos + length

            if ref_end > region_start and ref_pos < region_end:
                # overlaps with region
                if ref_pos < region_start:  # trim left <..[..>..] or <..[...]..>
                    length -= region_start - ref_pos

                if ref_end > region_end:  # trim right [..<..]..> or <..[...]..>
                    length -= ref_end - region_end

                trimmed_cigar.append((op, length))

            ref_pos = ref_end

        elif op in (1, 4, 5):  # I, S, H: consumes query only
            if ref_pos >= region_start and ref_pos < region_end:
                trimmed_cigar.append((op, length))

        elif op in (6, 9):  # P, B: consumes neither reference nor query
            continue

        else:
            raise ValueError(f"unsupported CIGAR operation: {op}")

    return trimmed_cigar


def parse_mdtag(
    segment: pysam.AlignedSegment,
) -> list[tuple[str, int]]:
    """Parse the MD tag of a pysam segment into a list of operations."""

    if not segment.has_tag("MD"):
        return list()

    md_pattern = re.compile(r"(\d+)|(\^[A-Z]+)|([A-Z])")
    md_string = segment.get_tag("MD")

    md_tokens = list()
    for match in md_pattern.finditer(md_string):
        if match.group(1):  # number of matches
            md_tokens.append(("=", int(match.group(1))))
        elif match.group(2):  # deletion
            deletion_seq = match.group(2)[1:]  # remove '^'
            md_tokens.append(("D", len(deletion_seq)))
        elif match.group(3):  # mismatch
            md_tokens.append(("X", 1))

    return md_tokens


def trim_mdtag(
    segment: pysam.AlignedSegment,
    region_start: int,
    region_end: int,
) -> list[tuple[str, int]]:
    """Trim MD tag operations to those overlapping the specified reference region."""

    md_tokens = parse_mdtag(segment)
    if not md_tokens:
        return list()

    ref_pos = segment.reference_start
    trimmed_md = list()

    for op, length in md_tokens:
        if op in ("=", "X", "D"):  # consumes reference
            ref_end = ref_pos + length

            if ref_end > region_start and ref_pos < region_end:
                # overlaps with region
                if ref_pos < region_start:  # trim left <..[..>..] or <..[...]..>
                    length -= region_start - ref_pos

                if ref_end > region_end:  # trim right [..<..]..> or <..[...]..>
                    length -= ref_end - region_end

                trimmed_md.append((op, length))

            ref_pos = ref_end

        else:
            raise ValueError(f"unsupported MD operation: {op}")

    return trimmed_md


def update_feature_matrix(
    feature_mat: np.ndarray,
    trimmed_cigar: list[tuple[int, int]],
    trimmed_md: list[tuple[str, int]],
    related_start: int,
    related_end: int,
) -> None:
    """Update the feature matrix using trimmed CIGAR and MD operations over a region."""
    #      0              1              2               3               4             5             6            7         8
    # MISMATCHCOUNT, DELETIONCOUNT, SOFTHARDCOUNT, INSERTIONCOUNT, INSERTIONMEAN, INSERTIONMAX, DELETIONMEAN, DELETIONMAX, DEPTH

    feature_mat[related_start:related_end, 8] += 1  # depth

    start = related_start

    for op, length in trimmed_cigar:
        if (
            op
            in (
                0,
                2,
                3,
                7,
                8,
            )
        ):  # M, D, N, =, X: consumes reference (M, D, N, =, X) and possibly query (M, =, X)
            end = start + length
        elif op in (1, 4, 5):  # I, S, H: consumes query only
            end = start
        else:
            raise ValueError(f"Unsupported CIGAR operation: {op}")

        if op == 2:  # deletion
            feature_mat[start:end, 1] += 1  # deletion count
            feature_mat[start:end, 6] += length  # deletion mean (will divide later)
            feature_mat[start:end, 7] = np.maximum(
                feature_mat[start:end, 7], length
            )  # deletion max

        elif op == 1:  # insertion
            feature_mat[start, 3] += 1  # insertion count
            feature_mat[start, 4] += length  # insertion mean (will divide later)
            feature_mat[start, 5] = np.maximum(
                feature_mat[start, 5], length
            )  # insertion max

        elif op in (4, 5):  # soft clip or hard clip
            feature_mat[start, 2] += 1  # soft hard count

        start = end

    start = related_start  # reset start for MD tag processing
    for op, length in trimmed_md:
        if op in ("=", "X", "D"):  # consumes reference
            end = start + length
        else:
            raise ValueError(f"Unsupported MD operation: {op}")

        if op == "X":  # mismatch
            feature_mat[start:end, 0] += 1  # mismatch count

        start = end


def main(
    bam_file: str,
    contig: str,
    region_start: int,
    region_end: int,
    window_size: int = 200,
    output_directory: str = "./features/",
) -> None:
    """Process a BAM file and generate MAMNET feature matrices for a genomic region."""

    os.makedirs(output_directory, exist_ok=True)

    try:
        bam = pysam.AlignmentFile(bam_file, "rb")
    except Exception as e:
        raise RuntimeError(f"error opening bam file: {e}") from e

    region_length = region_end - region_start
    feature_matrix = np.zeros(
        (region_length, 9), dtype=np.float16
    )  # 9 features as defined

    for segment in bam.fetch(contig, region_start, region_end):
        if segment.is_unmapped:
            continue

        trimmed_cigar = trim_cigar(segment, region_start, region_end)
        trimmed_md = trim_mdtag(segment, region_start, region_end)

        related_start = max(0, segment.reference_start - region_start)
        related_end = min(region_length, segment.reference_end - region_start)

        update_feature_matrix(
            feature_matrix, trimmed_cigar, trimmed_md, related_start, related_end
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        insertion_counts = feature_matrix[:, 3]
        deletion_counts = feature_matrix[:, 1]

        feature_matrix[:, 4] = np.where(
            insertion_counts > 0, feature_matrix[:, 4] / insertion_counts, 0
        )  # insertion mean
        feature_matrix[:, 6] = np.where(
            deletion_counts > 0, feature_matrix[:, 6] / deletion_counts, 0
        )  # deletion mean

    for start in list(range(0, region_length + 1, window_size))[
        :-1
    ]:  # exclude last window if smaller than window size
        end = start + window_size
        window_features = feature_matrix[start:end]

        if np.sum(window_features) == 0:
            continue

        normalized_features = logify_numpy(window_features).astype("float16")

        window_index = region_start + start
        output_path = os.path.join(
            output_directory,
            f"{contig}_{window_index}_{window_index + (end - start)}.npy",
        )

        np.save(output_path, normalized_features)
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(
                normalized_features.T,
                cmap="viridis",
                ax=ax,
                cbar_kws={"label": "log-normalized feature value"},
                yticklabels=[
                    "MISMATCHCOUNT",
                    "DELETIONCOUNT",
                    "SOFTHARDCOUNT",
                    "INSERTIONCOUNT",
                    "INSERTIONMEAN",
                    "INSERTIONMAX",
                    "DELETIONMEAN",
                    "DELETIONMAX",
                    "DEPTH",
                ],
            )
            ax.set_xlabel("position in window")
            ax.set_title(
                f"features heatmap: {contig}:{window_index}-{window_index + (end - start)}"
            )
            fig.tight_layout()
            plt_path = os.path.join(
                output_directory,
                f"{contig}_{window_index}_{window_index + (end - start)}.png",
            )
            fig.savefig(plt_path, dpi=300)
            plt.close(fig)
        except ImportError:
            print("seaborn or matplotlib not installed, skipping heatmap generation.")

    bam.close()


def parse_arguments() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="MAMNET-style feature extraction from BAM alignments")

    parser.add_argument("-b", "--bam", default="data/HG002_chr21.bam", help="input BAM file (default: data/HG002_chr21.bam)")
    parser.add_argument("-c", "--contig", default="chr21", help="chromosome / contig name (default: chr21)")
    parser.add_argument("-s", "--start", type=int, default=11019054, help="start position (0-based, default: 11019054)")
    parser.add_argument("-e", "--end", type=int, default=11020031, help="end position (0-based, exclusive; default: 11020031)")
    parser.add_argument("-w", "--window-size", type=int, default=200, help="window size for feature tiling (default: 200)")
    parser.add_argument("-o", "--output-directory", default="output/features", help="output directory for feature matrices and plots (default: output/features)")
    # fmt: on

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    main(
        bam_file=args.bam,
        contig=args.contig,
        region_start=args.start,
        region_end=args.end,
        window_size=args.window_size,
        output_directory=args.output_directory,
    )
