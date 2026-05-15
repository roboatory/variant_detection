import pysam
#import pybedtools
import numpy as np
from collections import defaultdict
# import pandas as pd
from encode_read_alignment import encode_region
from multiprocessing.pool import Pool
import os
import tqdm

pysam.set_verbosity(0) # temporary fix t supress pysam bai older than bam warnings

def get_chrome_len(fai_path):
    genome = {}
    with open(fai_path, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            genome[fields[0]] = int(fields[1])
    return genome

def extract_sv_regions(vcf_path):
    sv_regions = defaultdict(list)
    vcf = pysam.VariantFile(vcf_path)
    for record in vcf.fetch():
        chrom = record.chrom
        start = record.pos
        end = record.stop if record.stop else record.pos + 1
        sv_regions[chrom].append((start, end))

    for chrom in sv_regions.keys():
        sv_regions[chrom] = sorted(sv_regions[chrom], key=lambda x: (x[0], x[1]))

    return sv_regions

def cluster_svs(sv_regions, chrome_lens, distance_threshold=1999):
    clustered_regions = defaultdict(list)

    for chrom, regions in sv_regions.items():
        if not regions:
            continue

        chrom_len = chrome_lens[chrom]

        current_start, current_end = regions[0]
        svs_in_cluster = [regions[0]]


        for start, end in regions[1:]:
            if start - current_end <= distance_threshold:
                current_end = max(current_end, end)
                svs_in_cluster.append((start, end))
            else:
                clustered_regions[chrom].append([(max(0, current_start-distance_threshold), min(current_end+distance_threshold, chrom_len)), svs_in_cluster])
                current_start, current_end = start, end
                svs_in_cluster = [(start, end)]

        clustered_regions[chrom].append([(max(0, current_start-distance_threshold), min(current_end+distance_threshold, chrom_len)), svs_in_cluster])

    return clustered_regions


def sample_non_sv_windows(clustered_sv_regions, chrome_lens, window_size=2000, sample_per_chrom=1000):
    non_sv_windows = defaultdict(list)

    # for chrom, chrom_len in chrome_lens.items():
    for chrom, sv_clusters in clustered_sv_regions.items():
        
        chrom_len = chrome_lens[chrom]
        
        # sv_clusters = clustered_sv_regions.get(chrom, [])

        non_sv_intervals = list()

        for i in range(len(sv_clusters) + 1):
            if len(sv_clusters) == 0:
                non_sv_intervals.append((0, chrom_len))
            elif i == 0:
                if sv_clusters[0][0][0] == 0:
                    continue
                else:
                    non_sv_intervals.append((0, sv_clusters[0][0][0]))
            elif i == len(sv_clusters) and sv_clusters[-1][0][1] == chrom_len:
                continue
            elif i == len(sv_clusters):
                non_sv_intervals.append((sv_clusters[-1][0][1], chrom_len))
            else:
                non_sv_intervals.append((sv_clusters[i-1][0][1], sv_clusters[i][0][0]))

        # sample windows from random non-SV intervals
        while len(non_sv_windows[chrom]) < sample_per_chrom:
            sampled_intervals = np.random.choice(range(len(non_sv_intervals)), size=min(sample_per_chrom, len(non_sv_intervals)), replace=False)
            
            for i in sampled_intervals:
                interval_start, interval_end = non_sv_intervals[i]
                if interval_end - interval_start < window_size:
                    continue
                max_start = interval_end - window_size
                sampled_start = np.random.randint(interval_start, max_start + 1)
                non_sv_windows[chrom].append([(sampled_start, sampled_start + window_size),[]])

    return non_sv_windows


def check_bam_region(bam, chrom, start, end, sv_threshold=50):
    """
    Inspects reads in the window to ensure it is 'clean'.
    """
    try:
        iter_reads = bam.fetch(chrom, start, end)
    except ValueError:
        return False

    suspicious_signals = 0
    total_checked = 0

    for read in iter_reads:
        if read.is_unmapped or read.is_secondary or read.is_duplicate:
            continue

        total_checked += 1
        
        # 1. Check for Split Alignments (SA tag or supplementary flag)
        if read.is_supplementary or read.has_tag('SA'):
            # Only mark suspicious if the alignment boundary (breakpoint) falls within the window
            p_start = read.reference_start
            p_end = read.reference_end
            
            # Check if alignment start or end falls in [start, end)
            if (start <= p_start < end) or (start <= p_end < end):
                suspicious_signals += 1
                continue
            # If the split happens outside the window, we continue to check CIGAR for internal SVs

        current_ref_pos = read.reference_start
        has_local_sv = False

        if read.cigartuples:
            for op, length in read.cigartuples:
                consumes_ref = op in [0, 2, 3, 7, 8]
                
                if (op == 2 or op == 1) and length >= sv_threshold:
                    if op == 1: # INS
                        if start <= current_ref_pos < end:
                            has_local_sv = True
                    if op == 2: # DEL
                        del_start = current_ref_pos
                        del_end = current_ref_pos + length
                        if max(start, del_start) < min(end, del_end):
                            has_local_sv = True

                if has_local_sv:
                    suspicious_signals += 1
                    break 

                if consumes_ref:
                    current_ref_pos += length

    if total_checked > 0:
        ratio = suspicious_signals / total_checked
        if ratio > 0.10: 
            return False
    
    return True


def label_patches(chrom, region, svs, bam, patch_size=200,):

    num_patches = (region[1] - region[0]) // patch_size

    assert num_patches == 10 # ensure we have exactly 10 patches (2kbp) #NOTE: this is a temporary assertion for current design

    labels = np.zeros(num_patches, dtype=int)

    # for each patch, check if it overlaps with any SV
    for i in range(num_patches):
        patch_start = region[0] + i * patch_size
        patch_end = patch_start + patch_size

        for sv_start, sv_end in svs:
            # overlap at least 1bp
            if patch_end > sv_start and patch_start < sv_end:
                labels[i] = 1
                break
        # if not overlap with any SV, keep label 0 but double check in bam
        else:
            if not check_bam_region(bam, chrom, patch_start, patch_end):
                labels[i] = 1  # mark as SV suspicious (currently mark as 1, can be changed to another label if needed)

    return labels


def generate_labeled_windows(chrom, target_region, bam_file, output_dir, width=2000, stride=500, non_sv=False):

    bam = pysam.AlignmentFile(bam_file, "rb")

    feature_file_and_labels  = list()
    
    # for chrom, regions in target_regions.items():
    # for region_info in target_regions:
    region, svs = target_region
    # window slide with 500bp stride but need to cover the last base
    if (region[1] - region[0]) % width == 0:
        window_starts = list(range(region[0], region[1] - width + 1, stride))
        window_ends = list(range(region[0] + width, region[1] + 1, stride))
    else:
        window_starts = list(range(region[0], region[1] - width + 1, stride)) + [region[1] - width]
        window_ends = list(range(region[0] + width, region[1] + 1, stride)) + [region[1]]

    for win_start, win_end in zip(window_starts, window_ends):
        labels = label_patches(chrom, (win_start, win_end), svs, bam)

        if labels.sum() != 0 and non_sv:
            continue  # skip windows with any SV labels if generating non-SV data

        output_path = f"{output_dir}/{chrom}_{win_start}_{win_end}.npy"

        encoded_window = encode_region(bam, chrom, win_start, win_end)

        np.save(output_path, encoded_window)

        feature_file_and_labels.append((output_path, labels))

    return feature_file_and_labels



def main(bam_file, vcf, fai, output_dir, threads=4):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chrome_lens = get_chrome_len(fai)
    sv_regions = extract_sv_regions(vcf)
    clustered_sv_regions = cluster_svs(sv_regions, chrome_lens)
    non_sv_windows = sample_non_sv_windows(clustered_sv_regions, chrome_lens)

    record_f = open(output_dir + "/labels.txt", "w")
    
    with Pool(threads) as pool:
        # Process SV regions
        print("Processing SV regions...")
        results = []
        for chrom, regions in clustered_sv_regions.items():
            for region_info in regions:
                result = pool.apply_async(generate_labeled_windows, args=(chrom, region_info, bam_file, output_dir, 2000, 500, False))
                results.append(result)

        for result in tqdm.tqdm(results):
            feature_file_and_labels = result.get()
            for feature_file, labels in feature_file_and_labels:
                labels_str = ','.join(map(str, labels.tolist()))
                record_f.write(f"{feature_file}\t{labels_str}\n")

        # Process non-SV regions
        print("Processing non-SV regions...")
        results = []
        for chrom, regions in non_sv_windows.items():
            for region_info in regions:
                result = pool.apply_async(generate_labeled_windows, args=(chrom, region_info, bam_file, output_dir, 2000, 500, True))
                results.append(result)

        for result in tqdm.tqdm(results):
            feature_file_and_labels = result.get()
            for feature_file, labels in feature_file_and_labels:
                labels_str = ','.join(map(str, labels.tolist()))
                record_f.write(f"{feature_file}\t{labels_str}\n")

    record_f.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate labeled data for SV filtering")
    parser.add_argument("--bam", required=True, help="Input BAM file")
    parser.add_argument("--vcf", required=True, help="Input VCF file with SV calls")
    parser.add_argument("--fai", required=True, help="FAI index file for the reference genome")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output feature files and labels")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing")
    args = parser.parse_args()

    main(args.bam, args.vcf, args.fai, args.output_dir, args.threads)  