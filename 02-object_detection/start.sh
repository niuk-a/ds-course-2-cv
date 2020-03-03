#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

gcloud ai-platform jobs submit training object_detection_frsnn_`date +%m_%d_%Y_%H_%M_%S` \
    --job-dir=gs://02-object_detection/model_frcnn \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region us-east1 \
    --master-machine-type=standard_gpu \
    --runtime-version=1.15 \
    --scale-tier=CUSTOM \
    --worker-count=5 \
    --worker-machine-type=standard_gpu \
    --parameter-server-count=3 \
    --parameter-server-machine-type=standard \
    -- \
    --model_dir=gs://02-object_detection/model_frcnn \
    --pipeline_config_path=gs://02-object_detection/faster_rcnn/faster_rcnn_resnet101_coco.config


