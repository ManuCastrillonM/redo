import yaml
import os
import json

def yaml_to_dict(yml_path):
  with open(yml_path, 'r') as stream:

    try:
      params = yaml.load(stream)
    except yaml.YAMLError as exc:
      print(exc)

  return params
