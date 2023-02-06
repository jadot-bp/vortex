"""Module to convert openQCD format files to lime via lyncs-api."""

import sys

import lyncs_io as io


def main(input_file):
    """Reads openQCD format input file and converts to lime."""

    data = io.load(input_file, format="openqcd")

    io.lime.save(data, input_file + ".lime")


if __name__ == "__main__":
    main(sys.argv[1])
