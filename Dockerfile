#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
#FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

WORKDIR /app

RUN apt update -yqq && \
    apt install -y git libsndfile1 && \
    git clone https://github.com/Emotional-Text-to-Speech/pytorch-dc-tts && \
    git clone --recursive https://github.com/Emotional-Text-to-Speech/tacotron_pytorch.git && \
    cd "tacotron_pytorch/" && \
    pip install -e . && \
    pip install unidecode gdown librosa scikit-image inflect docopt tensorflow==1.15.4 matplotlib nltk protobuf==3.20.0 && \
    pip install typing-extensions numpy --upgrade

# download pth
RUN mkdir -p tacotron_pytorch/trained_models && \
    cd tacotron_pytorch/trained_models && \
    gdown -O angry_dctts.pth https://drive.google.com/uc?id=1o66wrcSUPK8j6LqO9iLTCTFTS83nKkrN && \
    gdown -O neutral_dctts.pth https://drive.google.com/uc?id=1vN1GOBz4C5j-1LTyi0GHNS59CjQXivS6 && \
    gdown -O ssrn.pth https://drive.google.com/uc?id=1xQ3KgOtsE5x58HtGYBo7SslQgFYQMj35 && \
    gdown -O amused_tacotron.pth https://drive.google.com/uc?id=1RWjVnpmD5lFdJYoByBIrPVJUr2m2I-ZC && \
    gdown -O sleepiness_tacotron.pth https://drive.google.com/uc?id=1jlwCl1E4H2Ds-_ics7D44jW78unKdm8S
