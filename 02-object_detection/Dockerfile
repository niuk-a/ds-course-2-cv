# FROM gcr.io/deeplearning-platform-release/tf-gpu.1-15
FROM gcr.io/deeplearning-platform-release/tf-cpu.1-15

RUN apt-get update --fix-missing && \
    apt-get install -y protobuf-compiler python-pil python-lxml python-tk git && \
    pip install --user Cython && \
    pip install --user contextlib2 && \
    pip install --user jupyter && \
    pip install --user matplotlib

WORKDIR /object_detection

RUN git clone https://github.com/tensorflow/models.git

RUN mkdir protoc && \
    wget -O protoc/protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip && \
    cd protoc && \
    unzip protobuf.zip

RUN cd models/research && \
    ../../protoc/bin/protoc object_detection/protos/*.proto --python_out=.

RUN git clone https://github.com/cocodataset/cocoapi.git
RUN cd cocoapi/PythonAPI && \
    make
RUN cp -r cocoapi/PythonAPI/pycocotools models/research/

WORKDIR models/research

CMD export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim && \
    python object_detection/builders/model_builder_test.py
