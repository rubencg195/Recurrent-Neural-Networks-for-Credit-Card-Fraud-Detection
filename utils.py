import os
import pandas as pd
import numpy as np

def setupTF(tf):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            # tf.config.gpu.set_per_process_memory_fraction(0.4)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("NO GPUs available")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

def testTry(fn, addErrMsg=""):
    try:
        fn()
    except Exception as e:
        print("Error:", addErrMsg,"-", repr(e))
        
def runCommand(cmd):
    try:
        stream = os.popen(cmd)
        output = stream.read().strip()
    except Exception as e:
        output = str(e).strip()
    print("CMD: {}\nOUT: {}\n{}".format(cmd, output, 10*"-"))
    return output

def runCommands(cmds):
    cmds_outputs = list()
    for cmd in cmds:
        try:
            output, stderr = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE).communicate()
            output = output.decode("utf-8").strip()
        except Exception as e:
            output = str(e).strip()
        print("CMD: {}\nOUT: {}\n{}".format(cmd, output, 10*"-"))
        cmds_outputs.append(output)     
    return cmds_outputs

def downloadFromKaggle(
    api_token = {"username":"rubencg195","key":"1a0667935c03c900bf8cc3b4538fa671"},
    kaggle_file_path='/home/ec2-user/.kaggle/kaggle.json',
    zip_file_path = "banksim1.zip"
    ):
    runCommands([
        "rm -rf "+str(Path(kaggle_file_path).parent),
        "mkdir -p "+str(Path(kaggle_file_path).parent)
    ])
    with open(kaggle_file_path, 'w+') as file:
        json.dump(api_token, file)
    runCommands([
        "chmod 600 "+kaggle_file_path,
        "kaggle datasets download -d ntnu-testimon/banksim1 --force"
    ])
    zip_ref = zipfile.ZipFile(zip_file_path, 'r')
    zip_ref.extractall()
    zip_ref.close()
    runCommand("ls *.csv")