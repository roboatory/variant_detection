import os
import gzip

import pysam


def read_fai(fai):
    fai_dict = dict()
    with open(fai, "r") as f:
        for line in f:
            if line[0] != "#":
                line = line.rstrip("\n").split("\t")
                fai_dict[line[0]] = [int(i) for i in line[1:]]
    return fai_dict


def pos_convert(p, sinf, bpl, cpl):
    return sinf + (p // bpl) * cpl + p % bpl


def get_ref(ref, fai_dict, chrom, start, end=None):
    """
    0-based coordinates, [start, end)
    ref: file handle/object of the reference
    fai_dict: dict, key: chrom, value: [chrom_len, start_in_file, base_per_line, chr_per_line]
    """

    if end is None:
        end = fai_dict[chrom][0]

    if end > fai_dict[chrom][0]:
        raise ValueError("end point exceeds the chromosome length")
    if start > fai_dict[chrom][0]:
        raise ValueError("start point exceeds the chromosome length")
    if start > end:
        raise ValueError("start point larger than end point")

    s = pos_convert(start, *fai_dict[chrom][1:])
    e = pos_convert(end, *fai_dict[chrom][1:])

    ref.seek(s)
    seq = ref.read(e-s).replace("\n", "").upper()

    return seq


def infer_sv_type(ref, alts, info):
    sv_type = info.get("SVTYPE")
    if sv_type:
        return sv_type.upper()

    symbolic_types = {
        allele[1:-1].upper()
        for allele in alts
        if allele.startswith("<") and allele.endswith(">")
    }
    if len(symbolic_types) == 1:
        return next(iter(symbolic_types))

    if len(alts) == 1:
        if len(alts[0]) > len(ref):
            return "INS"
        if len(ref) > len(alts[0]):
            return "DEL"

    return None


def open_variant_file(vcf, mode="r", header=None):
    if header is None:
        return pysam.VariantFile(vcf, mode)
    return pysam.VariantFile(vcf, mode, header=header)


def open_text_auto(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def format_gt(sample_data):
    gt = sample_data.get("GT")
    if gt is None:
        return None

    alleles = ["." if allele is None else str(allele) for allele in gt]
    if len(alleles) <= 1:
        return alleles

    separator = "|" if sample_data.phased else "/"
    formatted_gt = []
    for index, allele in enumerate(alleles):
        if index > 0:
            formatted_gt.append(separator)
        formatted_gt.append(allele)
    return formatted_gt


def parse_pysam_vcf_record(vcf_record):
    alts = list(vcf_record.alts or [])
    info = dict(vcf_record.info)
    samples = list(vcf_record.samples)
    gt = None
    if samples:
        gt = format_gt(vcf_record.samples[samples[0]])

    return {
        "pysam_record": vcf_record,
        "chrom": vcf_record.chrom,
        "pos": vcf_record.pos,
        "id": vcf_record.id or ".",
        "ref": vcf_record.ref,
        "alt": ",".join(alts),
        "alts": alts,
        "qual": vcf_record.qual,
        "filter": list(vcf_record.filter.keys()),
        "info": info,
        "sv_type": infer_sv_type(vcf_record.ref, alts, info),
        "start": vcf_record.pos - 1,
        "end": vcf_record.pos - 1 + len(vcf_record.ref),
        "seqs": [vcf_record.ref] + alts,
        "samples": samples,
        "gt": gt,
        "is_header": False,
        "is_malformed": False,
    }


def SimpleVCFParser(vcf):
    if isinstance(vcf, (str, bytes, os.PathLike)):
        with open_variant_file(vcf, "r") as vcf_f:
            for vcf_record in vcf_f:
                yield parse_pysam_vcf_record(vcf_record)
    else:
        for vcf_record in vcf:
            yield parse_pysam_vcf_record(vcf_record)
