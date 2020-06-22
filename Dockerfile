FROM ubuntu:18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

RUN conda update -n base -c defaults conda

RUN mkdir git
WORKDIR git
RUN git clone --recurse-submodules https://github.com/pkyIntelligence/bertron.git

WORKDIR bertron
RUN conda env create -f environment.yaml --name bertron

# Pull the environment name out of the environment.yaml
RUN echo "source activate $(head -1 environment.yaml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yaml | cut -d' ' -f2)/bin:$PATH

RUN /bin/bash -c "source activate bertron && pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
RUN /bin/bash -c "source activate bertron && pip install git+https://github.com/flauted/coco-caption.git@python23"

WORKDIR ..
RUN git clone https://github.com/NVIDIA/apex
WORKDIR apex
RUN /bin/bash -c "source activate bertron && pip install -v --no-cache-dir --global-option=\"--pyprof\" --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\" ./"
WORKDIR ..

WORKDIR bertron/detectron2
RUN /bin/bash -c "source activate bertron && pip install -e ."
WORKDIR ../..

WORKDIR bertron/tacotron2
RUN sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt
WORKDIR ../..

RUN wget -O e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212014&authkey=AAHgqN3Y-LXcBvU"
RUN wget -O coco_g4_lr1e-6_batch64_scst.tar.gz "https://onedrive.live.com/download?cid=E5364FD183A1F5BB&resid=E5364FD183A1F5BB%212027&authkey=ACM1UXlFxgfWyt0"
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA" -O tacotron2_statedict.pt && rm -rf /tmp/cookies.txt
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx" -O waveglow_256channels_ljs_v2.pt && rm -rf /tmp/cookies.txt

RUN mv e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp_cpu.pkl model_weights/detectron
RUN tar -xf coco_g4_lr1e-6_batch64_scst.tar.gz
RUN mv coco_g4_lr1e-6_batch64_scst/model.19.bin model_weights/bert
RUN rm -rf coco_g4_lr1e-6_batch64_scst*
RUN mv tacotron2_statedict.pt model_weights/tacotron2
RUN /bin/bash -c "source activate bertron && python tacotron2/waveglow/convert_model.py waveglow_256channels_ljs_v2.pt model_weights/waveglow/fused_wg256ch_statedict.pt"
RUN rm waveglow_256channels_ljs_v2.pt

ENTRYPOINT ["conda", "run", "-n", "bertron", "python", "app.py", "config.json"]

