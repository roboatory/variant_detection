import argparse
import ast
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

matplotlib.use("agg")


def generate_index_file(fragments):
    output_file = fragments.replace("txt", "index")

    with open(fragments, "r") as source, open(output_file, "w") as destination:
        buffer = ("", 0)

        while signature := source.readline():
            read = signature.split("\t")[0]

            if read != "*":
                offset = source.tell() - len(signature)

                if buffer[0] != "":
                    destination.write(
                        "{}\t{}\t{}\n".format(buffer[0], buffer[1], offset)
                    )

                buffer = (read, offset)

        destination.write("{}\t{}\t{}\n".format(buffer[0], buffer[1], source.tell()))


def gather_SFS_signatures(bed, fragments, offsets, signatures, extension=50):
    destination = os.path.join(fragments, os.path.basename(bed))

    variant = os.path.splitext(os.path.basename(bed))[0].split("_")
    variant_start_position = int(variant[1])
    variant_end_position = int(variant[2])

    with (
        open(bed, "r") as bed_file,
        open(signatures, "r") as signatures_file,
        open(destination, "w") as fragments_file,
    ):
        _ = bed_file.readline()  # ignore BED header information

        fragments_file.write(
            "{}\t{}\t{}\t{}\t{}\n".format("CHROMOSOME", "START", "END", "READ", "TYPE")
        )  # write header to fragments file

        reads_encountered = set()

        for read in bed_file:
            read = read.rstrip().split("\t")

            if read[3] not in reads_encountered:
                start_offset = offsets.loc[read[3], "start"]
                end_offset = offsets.loc[read[3], "end"]

                signatures_file.seek(start_offset)

                for read_fragment in signatures_file.read(
                    end_offset - start_offset
                ).split("\n"):
                    tuples = [
                        ast.literal_eval(expression)
                        for expression in re.findall(r"\([^()]*\)", read_fragment)
                    ]

                    if len(tuples) >= 2:
                        for reference_tuple in tuples[
                            1:
                        ]:  # of the form (chromosome, reference_start, reference_end)
                            if (
                                all(reference_tuple) and reference_tuple[0] == read[0]
                            ):  # do chromosome locations match?
                                if max(
                                    reference_tuple[1],
                                    variant_start_position - extension,
                                ) <= min(
                                    reference_tuple[2], variant_end_position + extension
                                ):  # only capture overlapping fragments (+/- extension)
                                    fragments_file.write(
                                        "{}\t{}\t{}\t{}\t{}\n".format(
                                            read[0],
                                            reference_tuple[1],
                                            reference_tuple[2],
                                            read[3],
                                            "SFS",
                                        )
                                    )

                reads_encountered.add(read[3])


def visualize_fragments(fragments_file, images):
    variant = os.path.splitext(os.path.basename(fragments_file))[0].split("_")

    chromosome = variant[0]
    variant_start_position = int(variant[1])
    variant_end_position = int(variant[2])

    fragments = pd.read_csv(fragments_file, sep="\t")
    fragments["HEIGHT"] = fragments.groupby("READ").ngroup() + 2
    fragments["LENGTH"] = fragments["END"] - fragments["START"]

    def plot_signature_alignment():
        plt.plot([variant_start_position, variant_end_position], [1, 1], color="blue")
        for _, row in fragments.iterrows():
            plt.plot(
                [row["START"], row["END"]], [row["HEIGHT"], row["HEIGHT"]], color="red"
            )

        plt.xticks(rotation="vertical")
        plt.tick_params(left=False, labelleft=False)
        plt.ticklabel_format(axis="x", style="sci", useOffset=False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                images,
                "signatures/{}_{}_{}.png".format(
                    chromosome, variant_start_position, variant_end_position
                ),
            )
        )
        plt.clf()

    def plot_SFS_distribution():
        fragments["LENGTH"].plot.hist()

        plt.xlabel("length")
        plt.ylabel("number of fragments")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                images,
                "distributions/{}_{}_{}.png".format(
                    chromosome, variant_start_position, variant_end_position
                ),
            )
        )
        plt.clf()

    plot_signature_alignment()
    plot_SFS_distribution()


def launch_chromosome_analysis(
    chromosome, base_bed, base_images, base_fragments, offsets, signatures
):
    bed = os.path.join(base_bed, chromosome)
    images = os.path.join(base_images, chromosome)
    fragments = os.path.join(base_fragments, chromosome)

    os.makedirs(os.path.join(images, "signatures"), exist_ok=True)
    os.makedirs(os.path.join(images, "distributions"), exist_ok=True)
    os.makedirs(fragments, exist_ok=True)

    for bed_file in os.listdir(bed):
        gather_SFS_signatures(
            os.path.join(bed, bed_file), fragments, offsets, signatures
        )
        visualize_fragments(os.path.join(fragments, bed_file), images)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="sample-specific string (SFS) analysis")

    parser.add_argument("-c", "--chromosomes", default="chr21", help="limits SFS analysis to particular chromosomes; specify as a comma separated list or using the keyword 'all' for the entire genome (default: chr21)")
    parser.add_argument("-d", "--bed", default="data/bed", help="path to variant BED directory (default: data/bed)")
    parser.add_argument("-f", "--fragments", default="data/fragments", help="output location for SFS binned by chromosomal variant (default: data/fragments)")
    parser.add_argument("-g", "--generate_index", action="store_true", help="generate a .index file for fast lookup")
    parser.add_argument("-i", "--images", default="data/fragment-images", help="output image directory (default: data/fragment-images)")
    parser.add_argument("-s", "--signatures", default="data/SFS_signatures.txt", help="path to .txt file containing the extracted SFS (default: data/SFS_signatures.txt)")
    # fmt: on

    return parser.parse_args()


def main():
    args = parse_args()
    chromosomes = (
        ["chr{}".format(chromosome_number) for chromosome_number in range(1, 23)]
        if args.chromosomes == "all"
        else args.chromosomes.split(",")
    )
    base_bed = args.bed
    base_fragments = args.fragments
    generate_index = args.generate_index
    base_images = args.images
    signatures = args.signatures

    if generate_index:
        generate_index_file(signatures)

    if os.path.exists(signatures.replace("txt", "index")):
        offsets = pd.read_csv(
            signatures.replace("txt", "index"),
            delimiter="\t",
            index_col=0,
            names=["start", "end"],
        )

        for chromosome in chromosomes:
            launch_chromosome_analysis(
                chromosome, base_bed, base_images, base_fragments, offsets, signatures
            )

    else:
        print("please generate a .index file by adding the flag --generate_index")


main()
