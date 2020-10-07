from flask import Flask, current_app
from bertron import Bertron
import json
import os
import torch


def create_app(app_root, config_path, device_str):
    """Create the app."""
    app = Flask(__name__, static_url_path='/static')

    with open(config_path) as f:
        config_json = json.load(f)

    sampling_rate = config_json["sampling_rate"]
    app.bertron = Bertron(detector_cfg_path=os.path.join(app_root, config_json["detectron_config_path"]),
                      detector_weights_path=os.path.join(app_root, config_json["detectron_weights_path"]),
                      bert_cfg_path=os.path.join(app_root, config_json["bert_config_path"]),
                      bert_weights_path=os.path.join(app_root, config_json["bert_weights_path"]),
                      object_vocab_path=os.path.join(app_root, config_json["object_vocab_path"]),
                      tacotron_weights_path=os.path.join(app_root, config_json["tacotron_weights_path"]),
                      waveglow_cfg_path=os.path.join(app_root, config_json["waveglow_config_path"]),
                      waveglow_weights_path=os.path.join(app_root, config_json["waveglow_weights_path"]),
                      cpu_device=torch.device("cpu"),
                      gpu_device=torch.device("cuda") if device_str == "gpu" else None,
                      sampling_rate=sampling_rate)
    app.sampling_rate = sampling_rate

    return app