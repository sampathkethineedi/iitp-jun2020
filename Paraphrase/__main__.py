import sys
import os
import csv

from .model import T5Model
from .file_parser import write_data, read_data, get_params


def main():
    args = sys.argv[1:]

    # No of arguments check
    if len(args) != 3:
        raise Exception("The input format should be <config> <inputfilename> <outputfilename>")

    configPath = args[0]
    inputPath = args[1]
    outputPath = args[2]

    data = read_data(inputPath)
    params = get_params(configPath)

    print("Initializing the T5 model")
    T5 = T5Model(params)

    for row in data:
        output = T5.forward(row)
        print(output)
        write_data(outputPath, output)


if __name__ == '__main__':
    main()
