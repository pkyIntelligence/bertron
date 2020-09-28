import torch
import torchaudio

from io import BytesIO
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import requests
import sys
import time
import validators

from bertron import Bertron

from flask import Flask, request, render_template

# Set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(APP_ROOT, 'config.json')
device_str = 'cpu'

with open(config_path) as f:
    config_json = json.load(f)

sampling_rate = config_json["sampling_rate"]
bertron = Bertron(detector_cfg_path=os.path.join(APP_ROOT, config_json["detectron_config_path"]),
                  detector_weights_path=os.path.join(APP_ROOT, config_json["detectron_weights_path"]),
                  bert_cfg_path=os.path.join(APP_ROOT, config_json["bert_config_path"]),
                  bert_weights_path=os.path.join(APP_ROOT, config_json["bert_weights_path"]),
                  object_vocab_path=os.path.join(APP_ROOT, config_json["object_vocab_path"]),
                  tacotron_weights_path=os.path.join(APP_ROOT, config_json["tacotron_weights_path"]),
                  waveglow_cfg_path=os.path.join(APP_ROOT, config_json["waveglow_config_path"]),
                  waveglow_weights_path=os.path.join(APP_ROOT, config_json["waveglow_weights_path"]),
                  cpu_device=torch.device("cpu"),
                  gpu_device=torch.device("cuda") if device_str == "gpu" else None,
                  sampling_rate=sampling_rate)

# Define Flask app
app = Flask(__name__, static_url_path='/static')


# Define apps home page
@app.route('/')
def index():
    return render_template('index.html', generated_audio=False)


# Define submit function
@app.route('/submit', methods=['POST'])
def submit():
    static_dir = os.path.join(APP_ROOT, "static/")

    if not os.path.isdir(static_dir):
        os.mkdir(static_dir)

    visualize = "visualize" in request.form
    denoise = "denoise" in request.form

    if request.form["top_n"] == "":
        top_n = 0
    else:
        top_n = int(request.form["top_n"])

    if top_n < 0:
        top_n = 0

    if top_n > 100:
        top_n = 100

    image_url = request.form["image_url"]
    if not validators.url(image_url):
        return render_template('index.html', generated_audio=False, invalid_url=True, visualize=visualize, top_n=top_n,
                               denoise=denoise, current_url=image_url)

    response = requests.get(image_url)

    if response.status_code != 200:
        return render_template('index.html', generated_audio=False, unsuccessful_request=True, visualize=visualize,
                               top_n=top_n, denoise=denoise, current_url=image_url)

    try:
        img = np.array(Image.open(BytesIO(response.content)))[:, :, ::-1]
    except:
        return render_template('index.html', generated_audio=False, non_image_url=True, visualize=visualize,
                               top_n=top_n, denoise=denoise, current_url=image_url)

    audio, vis_output, caption, mel_outputs, mel_outputs_postnet, alignments = \
        bertron(img, visualize=visualize, viz_top_n=top_n, denoise=denoise)

    Image.fromarray(vis_output).save(os.path.join(static_dir, "image.jpg"))

    mel_data = (mel_outputs.float().data.cpu().numpy()[0], mel_outputs_postnet.float().data.cpu().numpy()[0],
                alignments.float().data.cpu().numpy()[0].T)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.set_title("Mel Spectogram")
    ax.set_ylabel("Channel")
    ax.set_xlabel("Frames")
    ax.imshow(mel_data[0], origin="lower")
    fig.savefig(os.path.join(static_dir, "mel_outputs.png"))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.set_title("Mel Spectogram Post-Net")
    ax.set_ylabel("Channel")
    ax.set_xlabel("Frames")
    ax.imshow(mel_data[1], origin="lower")
    fig.savefig(os.path.join(static_dir, "mel_outputs_postnet.png"))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.set_title("Alignment (Attention Map)")
    ax.set_ylabel("Character Position")
    ax.set_xlabel("Frames")
    ax.imshow(mel_data[2], origin="lower")
    fig.savefig(os.path.join(static_dir, "alignments.png"))

    torchaudio.save(os.path.join(static_dir, "audio.wav"), audio.float().cpu(), sampling_rate)

    return render_template('index.html', generated_audio=True, now=time.time(), visualize=visualize, caption=caption,
                           current_url=image_url, top_n=top_n, denoise=denoise)


# Start the application
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
