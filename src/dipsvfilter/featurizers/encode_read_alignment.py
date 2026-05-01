import numpy as np
import pysam
import re
import os

def logify_numpy(a): #Credit to Rohit
    """logarithmic transformation used in mamnet normalization"""
    return (np.log(((a > 0) * a) + 1.) - np.log((np.abs(a) * (a < 0)) + 1.))
# ​log(a+1) a>0
# 0 a=0
# −log(∣a∣+1) a<0

def trim_cigar(segment, region_start, region_end):
    """
    For a given aligned segment and a target reference region,
    return the overlapping portion of the cigartuple.

    Parameters:
        segment (pysam.AlignedSegment): The aligned segment
        region_start (int): Start of the reference region (0-based)
        region_end (int): End of the reference region (0-based, exclusive)

    Returns:
        list of tuples: Trimmed cigartuple overlapping with the region
    """
    ref_pos = segment.reference_start

    trimmed_cigar = list()

    for op, length in segment.cigartuples:
        if op in (0, 2, 3, 7, 8):  # M, D, N, =, X: consumes ref (M, D, N, =, X) and possibly query (M, =, X)
            ref_end = ref_pos + length

            if ref_end > region_start and ref_pos < region_end:
                # Overlaps with region
                if ref_pos < region_start: # Trim left <..[..>..] or <..[...]..>
                    length -= (region_start - ref_pos)

                if ref_end > region_end: #` Trim right [..<..]..> or <..[...]..>
                    length -= (ref_end - region_end)

                trimmed_cigar.append((op, length))             

            ref_pos = ref_end

        elif op in (1, 4, 5):  # I, S, H: consumes query only
            if ref_pos >= region_start and ref_pos < region_end:
                trimmed_cigar.append((op, length))

        elif op in (6, 9):  # P, B: consumes neither ref nor query
            continue

        else:
            raise ValueError(f"Unsupported CIGAR operation: {op}")
        
    return trimmed_cigar


def parse_mdtag(segment):
    """
    Parse the MD tag of a pysam AlignedSegment into a list of operations.
    Parameters:
        segment (pysam.AlignedSegment): The aligned segment
    Returns:
        list of tuples: Parsed MD operations"""

    if not segment.has_tag('MD'):
        return list()
    
    MD_PATTERN = re.compile(r'(\d+)|(\^[A-Z]+)|([A-Z])')
    md_string = segment.get_tag('MD')

    md_tokens = list()
    for match in MD_PATTERN.finditer(md_string):
        if match.group(1):  # Number of matches
            md_tokens.append(('=', int(match.group(1))))
        elif match.group(2):  # Deletion
            deletion_seq = match.group(2)[1:]  # Remove '^'
            md_tokens.append(('D', len(deletion_seq)))
        elif match.group(3):  # Mismatch
            md_tokens.append(('X', 1))

    return md_tokens


def trim_mdtag(segment, region_start, region_end):
    """
    Trim the MD tag operations to the specified reference region.
    Parameters:
        segment (pysam.AlignedSegment): The aligned segment
        region_start (int): Start of the reference region (0-based)
        region_end (int): End of the reference region (0-based, exclusive)
    Returns:
        list of tuples: Trimmed MD operations overlapping with the region
    """

    md_tokens = parse_mdtag(segment)
    if not md_tokens:
        return list()

    ref_pos = segment.reference_start
    trimmed_md = list()

    for op, length in md_tokens:
        if op in ('=', 'X', 'D'):  # Consumes reference
            ref_end = ref_pos + length

            if ref_end > region_start and ref_pos < region_end:
                # Overlaps with region
                if ref_pos < region_start: # Trim left <..[..>..] or <..[...]..>
                    length -= (region_start - ref_pos)

                if ref_end > region_end: # Trim right [..<..]..> or <..[...]..>
                    length -= (ref_end - region_end)

                trimmed_md.append((op, length))             

            ref_pos = ref_end

        else:
            raise ValueError(f"Unsupported MD operation: {op}")

    return trimmed_md


#related_start = max(0, segment.reference_start - region_start)

def update_feature_matrix(feature_mat, trimmed_cigar, trimmed_md, related_start, related_end):
    """
    Update the feature matrix based on the trimmed CIGAR and MD operations.
    Parameters:
        feature_mat (np.ndarray): The feature matrix to update
        trimmed_cigar (list of tuples): Trimmed CIGAR operations
        trimmed_md (list of tuples): Trimmed MD operations
        related_start (int): Start index in the feature matrix corresponding to the reference region
        related_end (int): End index in the feature matrix corresponding to the reference region
    Returns:
        None (the feature_mat is updated in place)
    """
    #      0              1              2               3               4             5             6            7         8
    #MISMATCHCOUNT, DELETIONCOUNT, SOFTHARDCOUNT, INSERTIONCOUNT, INSERTIONMEAN, INSERTIONMAX, DELETIONMEAN, DELETIONMAX, DEPTH

    feature_mat[related_start:related_end, 8] += 1  # DEPTH

    start = related_start
    
    for op, length in trimmed_cigar:

        if op in (0, 2, 3, 7, 8): # M, D, N, =, X: consumes ref (M, D, N, =, X) and possibly query (M, =, X)
            end = start + length
        elif op in (1, 4, 5):  # I, S, H: consumes query only
            end = start
        else:
            raise ValueError(f"Unsupported CIGAR operation: {op}")
        
        if op == 2:  # Deletion
            feature_mat[start:end, 1] += 1  # DELETIONCOUNT
            feature_mat[start:end, 6] += length  # DELETIONMEAN (will divide later)
            feature_mat[start:end, 7] = np.maximum(feature_mat[start:end, 7], length)  # DELETIONMAX

        elif op == 1:  # Insertion
            feature_mat[start, 3] += 1  # INSERTIONCOUNT
            feature_mat[start, 4] += length  # INSERTIONMEAN (will divide later)
            feature_mat[start, 5] = np.maximum(feature_mat[start, 5], length)  # INSERTIONMAX

        elif op in (4, 5): # Soft clip or hard clip
            feature_mat[start, 2] += 1  # SOFTHARDCOUNT

        start = end

    start = related_start # reset start for MD tag processing
    for op, length in trimmed_md:
        if op in ('=', 'X', 'D'):  # Consumes reference
            end = start + length
        else:
            raise ValueError(f"Unsupported MD operation: {op}")

        if op == 'X':  # Mismatch
            feature_mat[start:end, 0] += 1  # MISMATCHCOUNT

        start = end


def encode_region(bam, contig, region_start, region_end):
    # initialize feature matrix
    region_length = region_end - region_start
    # num_windows = (region_length + window_size - 1) // window_size  # ceiling division
    feature_matrix = np.zeros((region_length, 9), dtype=np.float32) # 9 features as defined

    # iterate through reads in the specified region
    for segment in bam.fetch(contig, region_start, region_end):
        if segment.is_unmapped:
            continue

        # trim cigar and md tag to the region
        trimmed_cigar = trim_cigar(segment, region_start, region_end)
        trimmed_md = trim_mdtag(segment, region_start, region_end)

        # determine related start and end in the feature matrix
        related_start = max(0, segment.reference_start - region_start)
        related_end = min(region_length, segment.reference_end - region_start)

        # update feature matrix
        update_feature_matrix(feature_matrix, trimmed_cigar, trimmed_md, related_start, related_end)

    # finalize feature matrix (compute means)
    with np.errstate(divide='ignore', invalid='ignore'):
        insertion_counts = feature_matrix[:, 3]
        deletion_counts = feature_matrix[:, 1]

        feature_matrix[:, 4] = np.where(insertion_counts > 0, feature_matrix[:, 4] / insertion_counts, 0)  # INSERTIONMEAN
        feature_matrix[:, 6] = np.where(deletion_counts > 0, feature_matrix[:, 6] / deletion_counts, 0)    # DELETIONMEAN


    # apply log transformation (mamnet normalization)
    normalized_features = logify_numpy(feature_matrix).astype("float32")

    return normalized_features



def main(bam_file, contig, region_start, region_end, window_size=200, output_dir="./features/"):
    """
    Main function to process the BAM file and generate feature matrices for the specified region.
    Parameters:
        bam (str): Path to the sorted, indexed BAM file
        contig (str): Chromosome/contig name
        region_start (int): Start position of the region (0-based)
        region_end (int): End position of the region (0-based, exclusive)
        window_size (int): Size of the feature matrix window (default: 200)
        output_dir (str): Directory to save the feature matrices (default: "./features/")
    Returns:
        None (feature matrices are saved to files)"""
    
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # open bam file
    try:
        bam = pysam.AlignmentFile(bam_file, "rb")
    except Exception as e:
        print(f"error opening bam file: {e}")
        return None
    
    # initialize feature matrix
    region_length = region_end - region_start
    # num_windows = (region_length + window_size - 1) // window_size  # ceiling division
    feature_matrix = np.zeros((region_length, 9), dtype=np.float32) # 9 features as defined

    # iterate through reads in the specified region
    for segment in bam.fetch(contig, region_start, region_end):
        if segment.is_unmapped:
            continue

        # trim cigar and md tag to the region
        trimmed_cigar = trim_cigar(segment, region_start, region_end)
        trimmed_md = trim_mdtag(segment, region_start, region_end)

        # determine related start and end in the feature matrix
        related_start = max(0, segment.reference_start - region_start)
        related_end = min(region_length, segment.reference_end - region_start)

        # update feature matrix
        update_feature_matrix(feature_matrix, trimmed_cigar, trimmed_md, related_start, related_end)

    # finalize feature matrix (compute means)
    with np.errstate(divide='ignore', invalid='ignore'):
        insertion_counts = feature_matrix[:, 3]
        deletion_counts = feature_matrix[:, 1]

        feature_matrix[:, 4] = np.where(insertion_counts > 0, feature_matrix[:, 4] / insertion_counts, 0)  # INSERTIONMEAN
        feature_matrix[:, 6] = np.where(deletion_counts > 0, feature_matrix[:, 6] / deletion_counts, 0)    # DELETIONMEAN

    
    # TODO: more flexable segmentation 
    for start in list(range(0, region_length+1, window_size))[:-1]:  # exclude last window if smaller than window_size
        end = start + window_size
        window_features = feature_matrix[start:end]

        # filter out empty windows
        if np.sum(window_features) == 0:
            continue

        # apply log transformation (mamnet normalization)
        normalized_features = logify_numpy(window_features).astype("float32")

        # save to file
        window_index = region_start + start
        output_path = os.path.join(output_dir, f"{contig}_{window_index}_{window_index + (end - start)}.npy")
        np.save(output_path, normalized_features)

        # # plot as heatmap for quick visualization
        # try:
        #     import matplotlib.pyplot as plt
        #     plt.imshow(normalized_features.T, aspect='auto', cmap='hot', interpolation='nearest')
        #     plt.colorbar(label='Log-normalized feature value')
        #     plt.yticks(ticks=np.arange(9), labels=[
        #         'MISMATCHCOUNT', 'DELETIONCOUNT', 'SOFTHARDCOUNT', 'INSERTIONCOUNT',
        #         'INSERTIONMEAN', 'INSERTIONMAX', 'DELETIONMEAN', 'DELETIONMAX', 'DEPTH'
        #     ])
        #     plt.xlabel('Position in Window')
        #     plt.title(f'Features Heatmap: {contig}:{window_index}-{window_index + (end - start)}')
        #     plt.tight_layout()
        #     plt_path = os.path.join(output_dir, f"{contig}_{window_index}_{window_index + (end - start)}.png")
        #     plt.savefig(plt_path)
        #     plt.close()
        # except ImportError:
        #     print("matplotlib not installed, skipping heatmap generation.")

        # heatmap visualizaiton with seaborn
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 4))
            sns.heatmap(normalized_features.T, cmap='viridis', cbar_kws={'label': 'Log-normalized feature value'})
            plt.yticks(ticks=np.arange(9) + 0.5, labels=[
                'MISMATCHCOUNT', 'DELETIONCOUNT', 'SOFTHARDCOUNT', 'INSERTIONCOUNT',
                'INSERTIONMEAN', 'INSERTIONMAX', 'DELETIONMEAN', 'DELETIONMAX', 'DEPTH'
            ], rotation=0)
            plt.xlabel('Position in Window')
            plt.title(f'Features Heatmap: {contig}:{window_index}-{window_index + (end - start)}')
            plt.tight_layout()
            plt_path = os.path.join(output_dir, f"{contig}_{window_index}_{window_index + (end - start)}.png")
            plt.savefig(plt_path)
            plt.close()
        except ImportError:
            print("seaborn or matplotlib not installed, skipping heatmap generation.")    

    bam.close()

    
    



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='generate mamnet features from bam file')
    parser.add_argument('bam_file', help='path to sorted, indexed bam file')
    parser.add_argument('contig', help='chromosome/contig name')
    parser.add_argument('start', type=int, help='start position')
    parser.add_argument('end', type=int, help='end position')
    parser.add_argument("--window_size", type=int, default=200, help="window size (default: 200)")
    parser.add_argument("--output_dir", default="./features/", help="output directory (default: ./features/)")
    
    args = parser.parse_args()

    main(bam_file=args.bam_file,
        contig=args.contig,
        region_start=args.start,
        region_end=args.end,
        window_size=args.window_size,
        output_dir=args.output_dir)

    # main(bam_file="./HG002_CCS_chr21.bam",
    #     contig="chr21",
    #     region_start=34145006,
    #     region_end=34145206,
    #     window_size=200,
    #     output_dir="./features/")
    
    #32,916,574-32,916,772
    # main(bam_file="./HG002_CCS_chr21.bam",
    #     contig="chr21",
    #     region_start=32916570,
    #     region_end=32916776,
    #     window_size=200,
    #     output_dir="./features/")
    

    # 15,120,879-15,124,791
    # main(bam_file="./HG002_CCS_chr21.bam",
    #     contig="chr21",
    #     region_start=15120879,
    #     region_end=15124791,
    #     window_size=200,
    #     output_dir="./features/")
    

    # # 11,019,054-11,020,031
    # main(bam_file="./HG002_CCS_chr21.bam",
    #     contig="chr21",
    #     region_start=11019054,
    #     region_end=11020031,
    #     window_size=200,
    #     output_dir="./features/")