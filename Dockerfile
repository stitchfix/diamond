# Ubuntu image with R, python3 and diamond installed
FROM ubuntu

RUN apt-get update && \
apt-get install -y python3 gcc-4.9 tree python3-pip p7zip-full git r-base libssl-dev libcurl4-openssl-dev libssh2-1-dev vim

RUN Rscript -e "install.packages('devtools', repos = 'https://cran.cnr.berkeley.edu/')"
RUN Rscript -e "devtools::install_github('jyypma/nloptr')"
RUN Rscript -e "z <- lapply(c('ordinal', 'lme4', 'reshape2', 'logging', 'MCMCpack'), install.packages, repos = 'https://cran.cnr.berkeley.edu/')"
RUN pip3 install cython numpy pandas scipy jupyter seaborn nose future

# install diamond
RUN git clone http://github.com/stitchfix/diamond.git && pip3 install -e diamond
WORKDIR diamond/examples
CMD jupyter notebook --ip 0.0.0.0 --allow-root
