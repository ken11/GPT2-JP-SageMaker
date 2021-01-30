# Use AWS Deep Learning Containers Images for Base Image.
# Change the region etc. as needed.
# See the links below for more information on Deep Learning Containers Images.
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.3.1-gpu-py37-cu110-ubuntu18.04

COPY requirements.txt /opt/
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip --no-cache-dir install -r /opt/requirements.txt && \
    rm /opt/requirements.txt

COPY train.py /opt/ml/code/train.py
COPY load_tfrecord.py /opt/ml/code/load_tfrecord.py
COPY make_tfrecord.py /opt/ml/code/make_tfrecord.py

ENV TF_ENABLE_AUTO_MIXED_PRECISION 1
ENV SAGEMAKER_PROGRAM train.py
