#!/bin/bash

echo "Create environment"
pip3 install virtualenv
virtualenv verafin-mitacs-2020 -p python3.6  
source verafin-mitacs-2020/bin/activate 
python -V
pip install ipykernel

echo "Installing libraries"
pip install numpy shap scikit-learn sklearn-pandas imbalanced-learn
pip install keras Keras-Applications pydot graphviz pyparsing poetry
pip install isort black yapf ggplot PyAthena category_encoders pdpbox
pip install watermark si_prefix qgrid plotly lightgbm
pip install deap dask joblib xgboost boto3
pip install kaggle h5py progressbar2 seaborn matplotlib imbalanced-learn scipy s3fs
pip install tensorflow keras
pip install jupyter --upgrade

pip install jupyter_nbextensions_configurator jupyter_contrib_nbextensions
pip install jupyter_nbextensions_configurator
pip install qgrid
pip install jupyter --upgrade
    
jupyter contrib nbextensions install --user
jupyter nbextensions_configurator enable --user
jupyter nbextension enable --py --sys-prefix qgrid
jupyter nbextension enable --py --sys-prefix widgetsnbextension

python -m ipykernel install --user --name verafin-mitacs-2020 --display-name "Verafin MITACS 2020"  

deactivate

echo "End Installing libraries"