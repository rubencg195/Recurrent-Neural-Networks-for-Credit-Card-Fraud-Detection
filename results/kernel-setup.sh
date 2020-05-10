#!/bin/bash

    . /home/ec2-user/anaconda3/etc/profile.d/conda.sh
    if ! conda info --envs | grep -q verafin; then
        echo "Cloning Python3 environment"
        conda create --name verafin-mitacs-2020 --clone tensorflow_p36
        conda activate verafin-mitacs-2020
        pip install ipykernel
        python -m ipykernel install --user --name verafin-mitacs-2020 --display-name "Verafin MITACS 2020"
        conda deactivate
    fi

    echo "Installing libraries"
    conda activate verafin-mitacs-2020
    pip install numpy shap scikit-learn sklearn-pandas imbalanced-learn
    pip install keras Keras-Applications pydot graphviz pyparsing poetry
    pip install isort black yapf ggplot PyAthena category_encoders pdpbox
    pip install watermark si_prefix qgrid plotly lightgbm
    pip install deap dask joblib xgboost boto3 tensorflow==2.1.0 scikit-learn sklearn
    pip install kaggle h5py progressbar2 seaborn matplotlib imbalanced-learn scipy s3fs

    conda deactivate
    echo "Done installing libraries"

    echo "Installing jupyter nbextensions"
    source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv
    pip install jupyter_contrib_nbextensions
    pip install jupyter_nbextensions_configurator
    pip install qgrid
    
    jupyter contrib nbextensions install --user
    jupyter nbextensions_configurator enable --user
    jupyter nbextension enable --py --sys-prefix qgrid
    jupyter nbextension enable --py --sys-prefix widgetsnbextension
    
    jupyter nbextension enable collapsible_headings/main  
    jupyter nbextension enable code_prettify/main  
    jupyter nbextension enable codefolding/main  
    jupyter nbextension enable execute_time/main  
    jupyter nbextension enable ruler/main  
    jupyter nbextension enable table_beautifier/main  
    jupyter nbextension enable scratchpad/main  
    jupyter nbextension enable keyboard_shortcut_editor/main  
    jupyter nbextension enable toc2/main  


    echo "Restarting jupyter notebook server. Goodbye!"
    pkill -f jupyter-notebook
