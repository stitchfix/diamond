FROM jupyter/datascience-notebook

WORKDIR /diamond
ADD . /diamond
RUN pip2 install .
RUN cd examples
