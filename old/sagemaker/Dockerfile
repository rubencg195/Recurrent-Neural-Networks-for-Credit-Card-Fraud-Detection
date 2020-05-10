# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04
RUN sed -i'' 's/archive\.ubuntu\.com/us\.archive\.ubuntu\.com/' /etc/apt/sources.list
RUN apt-get update 
RUN apt-get install -y apt-transport-https
RUN apt-get install -y wget 
RUN apt-get install -y nginx
RUN apt-get install -y libgcc-5-dev
RUN apt-get install -y ca-certificates
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.6 python3-dev -y
RUN apt-get install build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip -y 
RUN rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN wget https://bootstrap.pypa.io/3.3/get-pip.py && python3 get-pip.py 
RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy scikit-learn xgboost pandas 
RUN pip3 install numpy shap scikit-learn sklearn-pandas imbalanced-learn  
RUN pip3 install keras Keras-Applications pydot graphviz pyparsing poetry 
RUN pip3 install isort yapf ggplot PyAthena category_encoders pdpbox 
RUN pip3 install watermark si_prefix qgrid plotly lightgbm flask gevent gunicorn 
RUN pip3 install deap dask joblib xgboost boto3 tensorflow-gpu ipython 
RUN pip3 install  kaggle h5py progressbar2 seaborn matplotlib imbalanced-learn scipy s3fs 
RUN pip3 install  jupyter --upgrade 

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY RNN /opt/program
# COPY DATA /opt/ml

WORKDIR /opt/program
