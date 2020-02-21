import torch

from captioner import Captioner
from tts import TTS


class Bertron:
    """
    A Captioner with a TTS Generator
    """
    def __init__(self, detector_cfg_path, detector_weights_path, bert_cfg_path, bert_weights_path, object_vocab_path,
                 tacotron_weights_path, waveglow_cfg_path, waveglow_weights_path, cpu_device, gpu_device, fc_layer=0,
                 max_caption_length=67, sampling_rate=22050):
        """
        args:
            detector_cfg_path: path to the detector config
            detector_weights_path: path to the detector weights
            bert_cfg_path: path to the bert decoder config
            bert_weights_path: path to the bert decoder weights
            tacotron_weights_path: path to the tacotron weights
            waveglow_weights_path: path to the waveglow weights
            cpu_device: The cpu device to run some parts of visualization
            gpu_device: The gpu device to run the bulk of computations, currently requires at least 1 GPU device
            fc_layer: the fully connected layer from the detector to extract features from, 0-indexed
            max_caption_length: the maximum number of tokens the caption can be
            sampling_rate: the rate that audio representations are sampled per second
        """
        self.captioner = Captioner(detector_cfg_path, detector_weights_path, bert_cfg_path, bert_weights_path,
                                   object_vocab_path, cpu_device, gpu_device, fc_layer, max_caption_length)

        self.tts = TTS(tacotron_weights_path, waveglow_cfg_path, waveglow_weights_path, sampling_rate)

    def __call__(self, img_path, visualize=False, viz_top_n=100, denoise=True):
        """
        inference only for now
        args:
            img_path: path or url to the image to caption
            visualize: Whether to display intermediary results, (detector output, text, mel spectograms)
            viz_top_n: how many top scoring detector's regions of interest to show
            denoise: whether to applying denoising to the final output
        """

        with torch.no_grad():

            text = self.captioner(img_path, visualize, viz_top_n)
            if visualize:
                print(text)
            return self.tts(text, visualize, denoise)
