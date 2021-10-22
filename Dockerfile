FROM circleci/python:3.7

RUN sudo apt-get update && sudo apt-get install libboost-dev libeigen3-dev
COPY . /project

WORKDIR /project
RUN sudo pip install --use-feature=in-tree-build .

CMD ["python"]
