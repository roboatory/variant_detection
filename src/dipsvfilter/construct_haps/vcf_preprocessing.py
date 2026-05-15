#!/usr/bin/env python3

"""Preprocess an SV VCF for haplotype construction.

This script keeps only INS and DEL records and regenerates the VCF ID column as:

    <chromosome>.<type>.<index>

For example:

    chr1.INS.1
    chr1.DEL.1

The index is counted separately for each chromosome and SV type.

TODO:
    Handle multiallelic and symbolic ALT records explicitly. For now, records
    with multiple ALT alleles or symbolic ALT alleles are skipped.
"""

from argparse import ArgumentParser
from collections import defaultdict
import sys

from utils import SimpleVCFParser, open_text_auto


ALLOWED_SV_TYPES = {"INS", "DEL"}


def has_symbolic_alt(alts):
    return any(alt.startswith("<") and alt.endswith(">") for alt in alts)


def preprocess_vcf(input_vcf, output_vcf):
    counts = defaultdict(int)
    kept = 0
    skipped = 0

    records = SimpleVCFParser(input_vcf)
    with open_text_auto(input_vcf, "rt") as in_f, open_text_auto(output_vcf, "wt") as out_f:
        for line in in_f:
            if line.startswith("#"):
                out_f.write(line)
                continue

            record = next(records)
            sv_type = record["sv_type"]

            if len(record["alts"]) != 1 or has_symbolic_alt(record["alts"]) or sv_type not in ALLOWED_SV_TYPES:
                skipped += 1
                continue

            chrom = record["chrom"]
            counts[(chrom, sv_type)] += 1
            columns = line.rstrip("\n").split("\t")
            columns[2] = f"{chrom}.{sv_type}.{counts[(chrom, sv_type)]}"
            out_f.write("\t".join(columns) + "\n")
            kept += 1

    return kept, skipped


def build_parser():
    parser = ArgumentParser(
        description="Keep INS/DEL records from a VCF and regenerate IDs as <chrom>.<type>.<index>."
    )
    parser.add_argument("--vcf", "-i", required=True, help="Input VCF")
    parser.add_argument("--out", "-o", required=True, help="Output preprocessed VCF")
    return parser


def main():
    args = build_parser().parse_args()
    kept, skipped = preprocess_vcf(args.vcf, args.out)
    print(f"Done. Kept {kept} INS/DEL records; skipped {skipped} other records.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
