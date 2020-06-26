# bertron

bertron is a project to create an end to end neural network which can analyze an image and describe it in a natural human voice. Please run the jupyter notebook for a demo: [BertronDemo](BertronDemo.ipynb)

![example](test_output/detector_output.png)

'A couple of people walking down a street with yellow umbrellas.'

![example_snd](test_output/mel1.png)

## Installation (with Docker)

currently the Dockerfile is CPU only for inference

Build the Dockerfile (Use sudo if you need to, this will take about 15 minutes depending on your CPU and network speed):
```
docker build -t <tag_name> https://github.com/pkyIntelligence/bertron.git  # Your tag can be anything
```

Run the server, name is not required but it makes looking up the ip address of the server easier:
```
docker run -it --name <container_name> <tag_name>
```

Depending on your exact docker configuration, figure out the ipaddress of the above container, for most people this will show it:
```
docker network inspect bridge  # Then locate your container via your container name above
```

Enter the ipaddress you see with port 5000 in your browser, it might be something like http://172.17.0.2:5000/

Inference takes about 10-15 seconds on a CPU. If you would like to install outside of a docker, please follow the steps in the Dockerfile and adjust for your own environment.

## Validation (For captions only)

### Data Setup

In order to validate we need validation data, we will be using Karpathy's split on COCO for validation.
I recommend having a central location for your data, mine is ~/Datasets, so all my coco data will be in ~/Datasets/coco, these steps assume you are in this folder.
Grab the data/annotations (38 GB Total):
```
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2014.zip
wget http://images.cocodataset.org/zips/test2015.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/annotations/image_info_test2014.zip
wget http://images.cocodataset.org/annotations/image_info_test2015.zip

wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
```

unzip the files
```
unzip '*.zip'
```

move the Karpathy split definition to annotations, you may delete the flickr ones for now, leaving them will do no harm:
```
mv dataset_coco.json annotations
rm dataset_flickr*  # optional
```

The list of coco images specifically for validation under the Karpathy split is included within this repository under coco/annotations/coco_valid_jpgs.json, copy it to the coco annotations folder:
```
export BERTRON_ROOT=~/git/bertron  # Change as appropriate for your setup
cp $BERTRON_ROOT/coco/annotations/coco_valid_jpgs.json annotations
```

Optionally, clean up the remaining zip files if you need the space
```
rm *.zip
```

That's it, assuming installation was also done correctly, activate the environment you installed bertron to, go to your bertron root and run the validate_coco_captions.py script (Takes about 30 minutes on an GeForce RTX 2080 Ti):
```
cd $BERTRON_ROOT
python validate_coco_captions.py \
    --detector_config detectron2/configs/COCO-Detection/faster_rcnn_X_101_64x4d_FPN_2x_vlp.yaml \
    --detector_weights model_weights/detectron/e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl \
    --decoder_config VLP/configs/bert_for_captioning.json \
    --decoder_weights model_weights/bert/model.19.bin \
    --object_vocab vocab/objects_vocab.txt \
    --coco_root ~/Datasets/coco \
    --coco_data_info annotations/dataset_coco.json \
    --coco_ann_file annotations/captions_val2014.json \
    --valid_jpgs_file annotations/coco_valid_jpgs.json \
    --batch_size 4 \
    --dl_workers 4
    
# Change to your coco root as approriate
# The model is quite large for GPU RAM, you may need to reduce batch_size if you run out of memory
# dl_workers hould at least be the number of cores your CPU has, if GPU util < 100%, try increasing
```

results are saved under $BERTRON_ROOT/eval_results/0_val.json

Current results with these weights are:
|        | BLEU 1 | BLEU 2 | BLEU 3 | BLEU 4 | METEOR | ROUGE_L | CIDEr | SPICE |
| ------ | ------ | ------ | ------ | ------ | ------ | ------- | ----- | ----- |
|Scores: | 0.654  | 0.478  | 0.338  | 0.237  | 0.230  | 0.494   | 0.763 | 0.159 |


## Acknowledgement
- Pretrained weights and most code started with https://github.com/LuoweiZhou/VLP
- Detectron2 for detector base: https://github.com/facebookresearch/detectron2
- Tacotron2 for pretrained weights and tts: https://github.com/NVIDIA/tacotron2
- Bottom-Up Top-Down Attention for general architecture and vocab: https://github.com/peteanderson80/bottom-up-attention
- bert and transformers: https://github.com/huggingface/transformers

And of course all the open source code and research that these all are built on as well.
