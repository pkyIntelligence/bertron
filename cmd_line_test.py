import torch
import torchaudio

from datetime import datetime
import logging
import os
import json
import sys

from bertron import Bertron

"""
Example usage:

python cmd_line_test.py <config_file_path> <image URL>
"""


def main(argv):

    if len(argv) < 3:
        print("Please specify a config file and an image URL like so: "
              "python cmd_line_test.py <config_file_path> <image URL>")
        return 1

    # Setting up logging
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    if not os.path.isdir("logging"):
        os.mkdir("logging")
    dt_format_string = "%Y-%m-%d_%H%M%S.%f"
    fh = logging.FileHandler(f"logging/{__file__}_{datetime.now().strftime(dt_format_string)}.log")
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    config_path = argv[1]
    image_URL = argv[2]

    with open(config_path) as f:
        config_json = json.load(f)
        logger.info("Configuration loaded")

    bertron = Bertron(detector_cfg_path=config_json["detectron_config_path"],
                      detector_weights_path=config_json["detectron_weights_path"],
                      bert_cfg_path=config_json["bert_config_path"],
                      bert_weights_path=config_json["bert_weights_path"],
                      object_vocab_path=config_json["object_vocab_path"],
                      tacotron_weights_path=config_json["tacotron_weights_path"],
                      waveglow_cfg_path=config_json["waveglow_config_path"],
                      waveglow_weights_path=config_json["waveglow_weights_path"],
                      cpu_device=torch.device("cpu"),
                      gpu_device=torch.device("cuda:0"),
                      sampling_rate=config_json["sampling_rate"])
    logger.info("Bertron created successfully")

    audio = bertron(image_URL)
    logger.info("Image description successfully converted to Audio")

    logger.info("Saving audio")
    torchaudio.save("cmd_line_output.wav", audio.cpu(), config_json["sampling_rate"])

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
