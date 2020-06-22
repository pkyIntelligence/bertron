FROM ubuntu:18.04

RUN apt update
RUN apt install -y git wget

# Anaconda installing
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
RUN bash Anaconda3-2020.02-Linux-x86_64.sh -b
RUN rm Anaconda3-2020.02-Linux-x86_64.sh

# Set path to conda
#ENV PATH /root/anaconda3/bin:$PATH
ENV PATH /home/ubuntu/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
RUN conda update anaconda
RUN conda update --all

RUN mkdir git
WORKDIR git
RUN git clone --recurse-submodules https://github.com/pkyIntelligence/bertron.git

WORKDIR bertron
RUN conda env create -f environment.yaml --name bertron

RUN wget -O e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212014&authkey=AAHgqN3Y-LXcBvU"
RUN wget -O coco_g4_lr1e-6_batch64_scst.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212027&authkey=ACM1UXlFxgfWyt0"
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA" -O tacotron2_statedict.pt && rm -rf /tmp/cookies.txt
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx" -O waveglow_256channels_ljs_v2.pt && rm -rf /tmp/cookies.txt

RUN mv e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp_cpu.pkl model_weights/detectron
RUN tar -xf coco_g4_lr1e-6_batch64_scst.tar.gz
RUN mv coco_g4_lr1e-6_batch64_scst/model.19.bin model_weights/bert
RUN rm -rf coco_g4_lr1e-6_batch64_scst*
RUN mv tacotron2_statedict.pt model_weights/tacotron2
RUN python tacotron2/waveglow/convert_model.py waveglow_256channels_ljs_v2.pt model_weights/waveglow/fused_wg256ch_statedict.pt
RUN rm waveglow_256channels_ljs_v2.pt

CMD python app.py config.json

