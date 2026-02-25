## Current `extract_cutesv_indels.py` Feature Extraction (Implementation Summary)

The script builds per-variant evidence from a BAM and VCF in two passes.  
First, it precomputes **split-read (inter-alignment) signatures** per chromosome by finding reads with multiple alignments and applying distance/overlap heuristics to adjacent segments.  
Then, for each VCF variant, it scans CIGAR operations in overlapping reads to collect **intra-alignment signatures** (`D` for deletions, `I` for insertions).  

For each variant, it writes a BED-like file with rows:
`CHROMOSOME, START, END, READ, TYPE` where `TYPE` is one of:
- `INTRA_DEL` / `INTRA_INS`
- `INTER_DEL` / `INTER_INS`

It also generates:
- a signature plot (variant interval plus read-support intervals), and
- an encoded matrix image where per-base support is represented as:
  - `0` = no support
  - `1` = intra-alignment support
  - `2` = inter-alignment support

### Concrete CIGAR Example (What Gets Recorded)

Assume:
- variant from VCF: `chr21:1005-1025` (so `start=1005`, `length=20`)
- extension used by the script: `50` bp
- one read: `readA`
- read alignment start: `990`
- simulated CIGAR: `10M6D20M4I10M`

The script tracks a reference cursor `current_read_position`, starting at `990`.

#### If `--type DEL`

Processing:
- `10M`: cursor `990 -> 1000`
- `6D`: deletion spans `1000-1006`; this overlaps the variant window (`955-1075`), so it writes:

```text
chr21    1000    1006    readA    INTRA_DEL
```

- `20M`: cursor `1006 -> 1026`
- `4I`: ignored in DEL mode
- `10M`: cursor `1026 -> 1036`

Result for this read in DEL mode: one `INTRA_DEL` record.

#### If `--type INS`

Processing:
- `10M`: cursor `990 -> 1000`
- `6D`: cursor `1000 -> 1006` (no INS record)
- `20M`: cursor `1006 -> 1026`
- `4I`: insertion tested at `1026-1030`; overlaps variant window, so it writes:

```text
chr21    1026    1030    readA    INTRA_INS
```

- `10M`: cursor continues from updated value

Result for this read in INS mode: one `INTRA_INS` record.

In both modes, any overlapping split-read signature found in the precomputed inter-alignment pass is appended as `INTER_DEL` or `INTER_INS` rows in the same per-variant BED file.
