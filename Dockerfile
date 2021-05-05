FROM jupyter/scipy-notebook

RUN mkdir my-model
ENV MODEL_DIR=/home/jovyan/my-model
ENV MODEL_FILE_LDA=clf_lda.joblib
ENV MODEL_FILE_NN=clf_nn.joblib

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt 

COPY train.csv ./train.csv
COPY test.csv ./test.csv

COPY train-lda.py ./train-lda.py
COPY train-nn.py ./train-nn.py
COPY train-auto-nn.py ./train-auto-nn.py

#RUN python3 train-lda.py
#RUN python3 train-nn.py
