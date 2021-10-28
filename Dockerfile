FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y libboost-dev libeigen3-dev
COPY . /project

WORKDIR /project
RUN pip install --use-feature=in-tree-build .

CMD ["python"]
