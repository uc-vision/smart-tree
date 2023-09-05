import torch


def load_model(model_path, weights_path, device=torch.device("cuda:0")):
    model = torch.load(f"{model_path}", map_location=device)
    model.load_state_dict(torch.load(f"{weights_path}"))
    model.eval()
    return model
