import sys
import os
import csv

from .model import T5Model
from .file_parser import get_writer, read_data, get_params


def main():
    print('in main and here')
    args = sys.argv[1:]

    configPath = args[0]
    inputPath = args[1]
    outputPath = args[2]

    data = read_data(inputPath)
    # print(data)

    params = get_params(configPath)

    T5 = T5Model(params)
    #
    for row in data:
        output = T5.forward(row)
        print(output)

    # No of arguments check
    # if len(args) != 3:
    #     raise Exception("The input format should be <config> <inputfilename> <outputfilename>")




    # print('count of args :: {}'.format(len(args)))
    # for arg in args:
    #     print('passed argument :: {}'.format(arg))

    # my_function('hello world')
    #
    # my_object = MyClass('Thomas')
    # my_object.say_name()


if __name__ == '__main__':
    main()
