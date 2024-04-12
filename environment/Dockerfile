# hash:sha256:4177086d42fcded8a19a1bed42e73136c79659d900642cf99210ed3fef7e7646
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=conda-forge --channel=defaults \
        dask==2022.9.1 \
        dask-image==2021.12.0 \
        fastparquet==0.4.1 \
        glob2==0.7 \
        pillow==9.4.0 \
        pyarrow==8.0.0 \
        python-spams==2.6.1 \
        scikit-image==0.19.2 \
        scikit-learn==1.1.3 \
    && conda clean -ya

COPY postInstall /
RUN /postInstall
