import csv
import json


def read_data(input_path):
    return_data = []
    with open(input_path, "r") as csvfile:
        data = csv.reader(csvfile)
        for row in data:
            return_data.append(row[0])

    return return_data


def get_writer(output_path):
    with open(output_path, "a") as csvfile:
        writer = csv.writer(csvfile)
    return writer


def get_params(config_path):
    with open(config_path, "r") as jsonfile:
        data = json.load(jsonfile)
        return data
