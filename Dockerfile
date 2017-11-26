# Ubuntu image with python3 and diamond installed
FROM ubuntu

RUN apt-get update && \
apt-get install -y python3 gcc-4.9 tree python3-pip p7zip-full git

RUN pip3 install cython numpy pandas scipy jupyter seaborn

# install diamond
RUN git clone http://github.com/stitchfix/diamond.git && pip3 install -e diamond
WORKDIR diamond/examples
CMD jupyter notebook --ip 0.0.0.0 --allow-root
