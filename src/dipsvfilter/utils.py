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

