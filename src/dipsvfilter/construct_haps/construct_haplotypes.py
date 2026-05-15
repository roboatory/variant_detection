# NOTE: We merge every SV into a cluster as long as they satisfy the distance threshold,
# and only check if they overlap with each other in the sequence generation step.
# 0/0 is included as a possible genotype state, haplotype labels are not recorded,
# and records with the same genotype combination are merged.
#
# This is originally the flank_SV.py
#
# changes:
# - Split FASTA header construction and haplotype sequence construction into
#   helper methods.
# - FASTA filenames record the shared cluster location as chrom_start-end.fasta.
# - FASTA headers use pipe-delimited fields for haplotype-specific information:
#   SVs=id1;id2|GT=gt1:gt2|hapSV=sv_id:start-end;sv_id:start-end.
#   hapSV coordinates are 0-based, half-open positions on the constructed
#   sequence; all-reference haplotypes use hapSV=NA.

# TODO: add methods to handle sequence collisions (different SV genotypes, but representing/generating the same sequence)

import itertools
import os

from utils import SimpleVCFParser, get_ref, read_fai


class SVCluster:
    ref = None
    fai_dict = None

    def __init__(self, sv):
        if SVCluster.ref is None or SVCluster.fai_dict is None:
            raise ValueError("Missing reference file and/or fai dictionary for SVCluster class")

        self.chrom = sv["chrom"]
        self.ids = [sv["id"]]
        self.bps = [(sv["start"], sv["end"])]
        self.seqs = [sv["seqs"]]
        self.gts = [["0" if i == "." else i for i in sv["gt"]]]

    def add_sv(self, sv):
        self.ids.append(sv["id"])
        self.bps.append((sv["start"], sv["end"]))
        self.seqs.append(sv["seqs"])
        self.gts.append(["0" if i == "." else i for i in sv["gt"]])

    def find_overlaps(self, hap_gts):
        for hap_gt in hap_gts:
            end = -1
            for gt, bp in zip(hap_gt, self.bps):
                if gt != 0:
                    if bp[0] < end:
                        return True
                    end = bp[1]
        return False

    def get_gt_combos(self):
        # referred to https://github.com/Sentieon/hap-eval/blob/main/hap_eval/hap_eval.py line 254-316
        combos = []
        for gt in self.gts:
            allele_pair = tuple(int(a) for i, a in enumerate(gt) if i % 2 == 0)
            if all(a == allele_pair[0] for a in allele_pair):
                gt_comb = [allele_pair, (0, 0)]
            elif gt[1] == "/":
                gt_comb = list(itertools.permutations(allele_pair))
                gt_comb.append((0, 0))
            elif gt[1] == "|":
                gt_comb = [allele_pair, (0, 0)]
            else:
                gt_comb = []
            combos.append(gt_comb)
        return combos

    def cluster_bounds(self, flank):
        cluster_start = max(0, self.bps[0][0] - flank)
        cluster_end = sorted(self.bps, key=lambda x: x[1])[-1][-1] + flank
        chrom_len = SVCluster.fai_dict[self.chrom][0]
        return cluster_start, min(cluster_end, chrom_len)

    def make_header(self, hap_gt, sv_intervals):
        sv_ids = ";".join(self.ids)
        gt_name = ":".join(str(gt) for gt in hap_gt)
        if sv_intervals:
            interval_name = ";".join(
                f"{self.ids[sv_index]}:{seq_start}-{seq_end}"
                for sv_index, seq_start, seq_end in sv_intervals
            )
        else:
            interval_name = "NA"

        return f">SVs={sv_ids}|GT={gt_name}|hapSV={interval_name}\n"

    def build_haplotype_sequence(self, cluster_start, cluster_end, hap_gt):
        hap_bps = [self.bps[i] for i in range(len(hap_gt)) if hap_gt[i] != 0]
        hap_seqs = [self.seqs[i][hap_gt[i]] for i in range(len(hap_gt)) if hap_gt[i] != 0]
        hap_indexes = [i for i in range(len(hap_gt)) if hap_gt[i] != 0]

        if len(hap_bps) == 0:
            seq = get_ref(SVCluster.ref, SVCluster.fai_dict, self.chrom, cluster_start, cluster_end)
            return seq, []

        prefix = get_ref(SVCluster.ref, SVCluster.fai_dict, self.chrom, cluster_start, hap_bps[0][0])
        parts = [prefix]
        sv_intervals = []
        current_pos = len(prefix)

        for i, (sv_index, hap_bp, hap_seq) in enumerate(zip(hap_indexes, hap_bps, hap_seqs)):
            seq_start = current_pos
            parts.append(hap_seq)
            current_pos += len(hap_seq)
            sv_intervals.append((sv_index, seq_start, current_pos))

            if i < len(hap_bps) - 1:
                connect = get_ref(
                    SVCluster.ref,
                    SVCluster.fai_dict,
                    self.chrom,
                    hap_bp[1],
                    hap_bps[i + 1][0],
                )
                parts.append(connect)
                current_pos += len(connect)

        suffix = get_ref(SVCluster.ref, SVCluster.fai_dict, self.chrom, hap_bps[-1][1], cluster_end)
        parts.append(suffix)
        return "".join(parts), sv_intervals

    def write(self, out_dir, flank):
        cluster_start, cluster_end = self.cluster_bounds(flank)

        combos = self.get_gt_combos()
        n = 1
        for combo in combos:
            n *= len(combo)
        if n > 1024:
            print(
                "%s:%d-%d too many phasing combos %d"
                % (self.chrom, self.bps[0][0], self.bps[-1][1], n)
            )
            return

        valid_combs = []
        
        for combo in itertools.product(*combos):
            if not combo:
                continue

            hap_gts = list(zip(*combo))
            if self.find_overlaps(hap_gts):
                continue

            valid_combs += hap_gts

        written = False
        out_fa = os.path.join(out_dir, f"{self.chrom}_{cluster_start}-{cluster_end}.fasta")
        with open(out_fa, "w") as f:
            for hap_gt in sorted(set(valid_combs)):
                seq, sv_intervals = self.build_haplotype_sequence(cluster_start, cluster_end, hap_gt)
                f.write(self.make_header(hap_gt, sv_intervals))
                f.write(seq + "\n")
                written = True

        if not written:
            print("[WARNING]: No valid SV combination found in Cluster:")
            for bp, gt in zip(self.bps, self.gts):
                print("\t" + self.chrom + "\t" + str(bp) + "\t" + "".join(gt))
            os.remove(out_fa)


def construct_haplotypes(vcf, ref, flank, out_dir):
    fai = ref + ".fai"
    fai_dict = read_fai(fai)
    ref_f = open(ref, "r")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    SVCluster.ref = ref_f
    SVCluster.fai_dict = fai_dict

    vcfparser = SimpleVCFParser(vcf)

    for sv in vcfparser:
        if sv["gt"] is None:
            continue
        if sv["gt"][0] == "0" and sv["gt"][-1] == "0":
            continue
        chrom = sv["chrom"]
        last_end = sv["end"]
        svc = SVCluster(sv)
        break
    else:
        raise ValueError("No valid SVCluster found in input VCF")

    for sv in vcfparser:
        if sv["gt"] is None:
            continue
        if sv["gt"][0] == "0" and sv["gt"][-1] == "0":
            continue

        if sv["chrom"] != chrom:
            svc.write(out_dir, flank)
            chrom = sv["chrom"]
            last_end = sv["end"]
            svc = SVCluster(sv)
        elif sv["start"] - last_end < flank:
            svc.add_sv(sv)
            if sv["end"] > last_end:
                last_end = sv["end"]
        else:
            svc.write(out_dir, flank)
            last_end = sv["end"]
            svc = SVCluster(sv)

    svc.write(out_dir, flank)

    ref_f.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--vcf", "-i", help="SORTED VCF")
    parser.add_argument("--out", "-o", help="Output directory")
    parser.add_argument("--flank", "-f", default=5000, type=int, help="length of flanking reference sequence")
    parser.add_argument("--ref", "-r", help="reference file")

    args = parser.parse_args()

    construct_haplotypes(args.vcf, args.ref, args.flank, args.out)
