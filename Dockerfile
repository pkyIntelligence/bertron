FROM ubuntu:18.04

RUN apt update
RUN apt install -y build-essential git python3 python3-pip wget libsndfile1

RUN pip3 install --upgrade setuptools pip

RUN mkdir git
WORKDIR git
RUN git clone --recurse-submodules https://github.com/pkyIntelligence/bertron.git

WORKDIR bertron
RUN pip3 install -r requirements.txt

RUN pip3 install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
RUN pip3 install git+https://github.com/flauted/coco-caption.git@python23

WORKDIR ..
RUN git clone https://github.com/NVIDIA/apex
WORKDIR apex
RUN pip3 install -v --no-cache-dir ./su
WORKDIR ..

WORKDIR bertron/detectron2
RUN pip3 install -e .
WORKDIR ../..

WORKDIR bertron/tacotron2
RUN sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt
WORKDIR ..

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qQyaTBAUW8T4slkdO73ywfsUOxuJCmZI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qQyaTBAUW8T4slkdO73ywfsUOxuJCmZI" -O e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl && rm -rf /tmp/cookies.txt

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1S_PE0CZkq1TLlYji9bovboUYztVgAEzM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1S_PE0CZkq1TLlYji9bovboUYztVgAEzM" -O fc7_b.pkl && rm -rf /tmp/cookies.txt

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UN1KTWx6DLZ0jmvtZU8Qav6dfJpgxeP9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UN1KTWx6DLZ0jmvtZU8Qav6dfJpgxeP9" -O fc7_w.pkl && rm -rf /tmp/cookies.txt

RUN mv e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl model_weights/detectron
RUN mv fc7_* model_weights/detectron

# RUN wget -O e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212014&authkey=AAHgqN3Y-LXcBvU"
# RUN mv e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl model_weights/detectron

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15DfBnGdAjHs93brAs7roS6Wj7TkCNnzv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15DfBnGdAjHs93brAs7roS6Wj7TkCNnzv" -O model.19.bin && rm -rf /tmp/cookies.txt
RUN mv model.19.bin model_weights/bert

# RUN wget -O coco_g4_lr1e-6_batch64_scst.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212027&authkey=ACM1UXlFxgfWyt0"
# RUN tar -xf coco_g4_lr1e-6_batch64_scst.tar.gz
# RUN mv coco_g4_lr1e-6_batch64_scst/model.19.bin model_weights/bert
# RUN rm -rf coco_g4_lr1e-6_batch64_scst*

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA" -O tacotron2_statedict.pt && rm -rf /tmp/cookies.txt
RUN mv tacotron2_statedict.pt model_weights/tacotron2

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx" -O waveglow_256channels_ljs_v2.pt && rm -rf /tmp/cookies.txt
RUN /bin/bash -c "source activate bertron && python tacotron2/waveglow/convert_model.py waveglow_256channels_ljs_v2.pt model_weights/waveglow/fused_wg256ch_statedict.pt cpu"
RUN rm waveglow_256channels_ljs_v2.pt

