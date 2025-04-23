import yaml
import os

def yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return data


if __name__ == '__main__':
    yaml_path = "./argparse/blood_resnet18.yaml"
    print(yaml_data(yaml_path))

