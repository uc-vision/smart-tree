from typing import List, Dict, Any
import torch


def get_properties(data_type, property_names: List[str]) -> Dict[str, Any]:
    properties = {}

    for prop_name in property_names:
        if hasattr(data_type, prop_name):
            properties[prop_name] = getattr(data_type, prop_name)

    return properties


def cat_tensor_dict(tensor_dict: Dict[str, Any]) -> torch.tensor:
    if len(tensor_dict) == 0:
        return list(tensor_dict.values())[0]
    return torch.cat(list(tensor_dict.values()), dim=0)


def cat_tensor_properties(data_type, property_names: List[str]) -> Dict[str, Any]:
    properties = []

    for prop_name in property_names:
        if hasattr(data_type, prop_name):
            properties.append(getattr(data_type, prop_name))

    if len(properties) == 1:
        return properties[0]
    return torch.cat(properties, 1)
