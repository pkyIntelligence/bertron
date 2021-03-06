FROM continuumio/miniconda3

RUN apt update
RUN apt install -y build-essential git wget libsndfile1 libsm6 libxext6 libxrender-dev apache2
RUN apt-get install -y libapache2-mod-wsgi-py3 nano

RUN a2enmod wsgi

RUN conda update -n base -c defaults conda

RUN useradd -ms /bin/bash wsgi-user

USER wsgi-user

WORKDIR /home/wsgi-user
RUN mkdir git
WORKDIR git
RUN git clone --recurse-submodules https://github.com/pkyIntelligence/bertron.git
WORKDIR bertron
RUN conda env create -f bertron_env.yml
RUN mv apache/__init__.py .
RUN mv apache/*.wsgi .
RUN mkdir static

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qQyaTBAUW8T4slkdO73ywfsUOxuJCmZI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qQyaTBAUW8T4slkdO73ywfsUOxuJCmZI" -O e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl && rm -rf /tmp/cookies.txt

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1S_PE0CZkq1TLlYji9bovboUYztVgAEzM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1S_PE0CZkq1TLlYji9bovboUYztVgAEzM" -O fc7_b.pkl && rm -rf /tmp/cookies.txt

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UN1KTWx6DLZ0jmvtZU8Qav6dfJpgxeP9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UN1KTWx6DLZ0jmvtZU8Qav6dfJpgxeP9" -O fc7_w.pkl && rm -rf /tmp/cookies.txt

RUN mv e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl model_weights/detectron
RUN mv fc7_* model_weights/detectron

# RUN wget -O e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212014&authkey=AAHgqN3Y-LXcBvU"
# RUN mv e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl model_weights/detectron

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15DfBnGdAjHs93brAs7roS6Wj7TkCNnzv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15DfBnGdAjHs93brAs7roS6Wj7TkCNnzv" -O model.19.bin && rm -rf /tmp/cookies.txt
RUN mv model.19.bin model_weights/bert

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA" -O tacotron2_statedict.pt && rm -rf /tmp/cookies.txt
RUN mv tacotron2_statedict.pt model_weights/tacotron2

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OwxZ6YAIlnfGftcSK0a24fwD2XGhZfSi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OwxZ6YAIlnfGftcSK0a24fwD2XGhZfSi" -O fused_wg256ch_statedict.pt && rm -rf /tmp/cookies.txt
RUN mv fused_wg256ch_statedict.pt model_weights/waveglow

USER root

RUN mv apache/bertron.conf /etc/apache2/sites-available
RUN a2dissite 000-default
RUN a2ensite bertron

USER wsgi-user

SHELL ["conda", "run", "-n", "bertron", "/bin/bash", "-c"]

RUN pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
RUN pip install git+https://github.com/flauted/coco-caption.git@python23
WORKDIR ..

WORKDIR bertron
RUN python3 -m pip install -e detectron2
WORKDIR ..

WORKDIR bertron/tacotron2
RUN sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt
WORKDIR ..

USER root

EXPOSE 80

