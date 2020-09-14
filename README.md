# Required
Install git
Install Docker: https://docs.docker.com/get-docker/

If you have GPU:
*     Install Nvidia-driver
*     Install Nvidia-docker: https://github.com/NVIDIA/nvidia-docker

- all following commands will run under assumption that you are in project root
- If error occurs, please help us fix it. Thank you !
# Training phase
## Data gathering
Copy all image and json file to one folder inside this project, named "labelme_data"

## Create Docker environment
```bash
git clone https://github.com/tensorflow/models.git
cd models
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t od .
```

## Run docker and install necessary packages
```bash
docker run -it --runtime=nvidia -v $PWD:/home/tensorflow/DocumentRotation od  bash
cd models/research
# update pip
pip install -U pip
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

```bash
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```

## Convert annotation data to trainable data

```bash
cd ~/
# install labelme
git clone https://github.com/wkentaro/labelme.git
cd labelme
pip install .
# convert labelme annotation to Coco annotation
python examples/instance_segmentation/labelme2coco.py ~/DocumentRotation/labelme_data/ ~/DocumentRotation/labelme_annotation/ --labels  ~/DocumentRotation/labels.txt --noviz
# convert Coco annotation to Tensor Record
cd ~/DocumentRotation
export TRAIN_IMAGE_DIR=~/DocumentRotation/labelme_annotation/
export VAL_IMAGE_DIR=~/DocumentRotation/labelme_annotation/
export TEST_IMAGE_DIR=~/DocumentRotation/labelme_annotation/
export TRAIN_ANNOTATIONS_FILE=~/DocumentRotation/labelme_annotation/annotations.json
export VAL_ANNOTATIONS_FILE=~/DocumentRotation/labelme_annotation/annotations.json
export TESTDEV_ANNOTATIONS_FILE=~/DocumentRotation/labelme_annotation/annotations.json
export OUTPUT_DIR=~/DocumentRotation/tfr/
python ~/models/research/object_detection/dataset_tools/create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
```

## download pre-trained model
Download from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
Model name: Mask R-CNN Inception ResNet V2 1024x1024
Save path: <project root>/
Unzip to: <project root>/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8

## Training

```bash
cd ~/DocumentRotation
export PIPELINE_CONFIG_PATH=pipeline.config
export MODEL_DIR=results/
TF_FORCE_GPU_ALLOW_GROWTH=true  python ../models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr
```

## export inference model
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true  python ../models/research/object_detection/exporter_main_v2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_dir ~/DocumentRotation/results/ \
    --input_type image_tensor \
    --output_directory=inference/

```

# Test Phase

# run test with Mask RCNN
```bash
python mask_rcnn_rotation.py --image "labelme_annotation/JPEGImages/1 (57).jpg"
```

# run test with keypoints
```bash
python keypoint_rotation.py -t "images/phoi_cmnd.png" --image "labelme_annotation/JPEGImages/1 (78).jpg"
```
