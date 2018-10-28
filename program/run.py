#!/usr/bin/env python

# docker run --memory=8000MB --cpus 2 -it -v `pwd`/:/tmp/program -v `pwd`/data:/tmp/input ckcollab/codalab-legacy:py3
# For testing: python3 /tmp/program/run.py /tmp/input/ /tmp/output/ /tmp/program/

#############################
# ChaLearn AutoML2 challenge #
#############################

# Usage: python program_dir/run.py input_dir output_dir program_dir

# program_dir is the directory of this program

#
# The input directory input_dir contains 5 subdirectories named by dataset,
# including:
# 	dataname/dataname_feat.type          -- the feature type "Numerical", "Binary", or "Categorical" (Note: if this file is abscent, get the feature type from the dataname.info file)
# 	dataname/dataname_public.info        -- parameters of the data and task, including metric and time_budget
# 	dataname/dataname_test.data          -- training, validation and test data (solutions/target values are given for training data only)
# 	dataname/dataname_train.data
# 	dataname/dataname_train.solution
# 	dataname/dataname_valid.data
#
# The output directory will receive the predicted values (no subdirectories):
# 	dataname_valid.predict           
# 	dataname_test.predict
# We have 2 test sets named "valid" and "test", please provide predictions for both.
# 
# We implemented 2 classes:
#
# 1) DATA LOADING:
#    ------------
# Use/modify 
#                  D = DataManager(basename, input_dir, ...) 
# to load and preprocess data.
#     Missing values --
#       Our default method for replacing missing values is trivial: they are replaced by 0.
#       We also add extra indicator features where missing values occurred. This doubles the number of features.
#     Categorical variables --
#       The location of potential Categorical variable is indicated in D.feat_type.
#       NOTHING special is done about them in this sample code. 
#     Feature selection --
#       We only implemented an ad hoc feature selection filter efficient for the 
#       dorothea dataset to show that performance improves significantly 
#       with that filter. It takes effect only for binary classification problems with sparse
#       matrices as input and unbalanced classes.
#
# 2) LEARNING MACHINE:
#    ----------------
# Use/modify 
#                 M = MyAutoML(D.info, ...) 
# to create a model.
#     Number of base estimators --
#       Our models are ensembles. Adding more estimators may improve their accuracy.
#       Use M.model.n_estimators = num
#     Training --
#       M.fit(D.data['X_train'], D.data['Y_train'])
#       Fit the parameters and hyper-parameters (all inclusive!)
#       What we implemented hard-codes hyper-parameters, you probably want to
#       optimize them. Also, we made a somewhat arbitrary choice of models in
#       for the various types of data, just to give some baseline results.
#       You probably want to do better model selection and/or add your own models.
#     Testing --
#       Y_valid = M.predict(D.data['X_valid'])
#       Y_test = M.predict(D.data['X_test']) 
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Lukasz Romaszko April 2015
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013
# Last modifications Isabelle Guyon, November 2017

# =========================== BEGIN USER OPTIONS ==============================
# Verbose mode: 
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
############## 
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there are 5 datasets). 
# The code should keep track of time spent and NOT exceed the time limit 
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 1200 

# Maximum number of cycles, number of samples, and estimators
#############################################################
# Your training algorithm may be fast, so you may want to limit anyways the 
# number of points on your learning curve (this is on a log scale, so each 
# point uses twice as many time than the previous one.)
# The original code was modified to do only a small "time probing" followed
# by one single cycle. We can now also give a maximum number of estimators 
# (base learners).
max_cycle = 1 
max_estimators = 10
max_samples = float('Inf')

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "../"
default_input_dir = root_dir + "sample_data"
default_output_dir = root_dir + "AutoML2_sample_result_submission"
default_program_dir = root_dir + "AutoML2_sample_code_program"

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Version of the sample code
version = 5 

# General purpose functions
import time
import numpy as np
overall_start = time.time()         # <== Mark starting time
import os
from sys import argv, path
import datetime
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

###############################
# ML Freiburg Methods
###############################

def get_info_from_file(datadir, dataset):
    """
    Get all information {attribute = value} pairs from the public.info file
    :param datadir:
    :param dataset:
    :return:
    """
    dataset_path = os.path.join(datadir, dataset, dataset + '_public.info')

    if not os.path.exists(dataset_path):
        sys.stderr.write('Could not find info: %s\n' % dataset_path)
        return {'time_budget': int(os.environ.get('TIME_LIMIT', 5*60))}

    info = dict()

    with open(dataset_path, 'r') as info_file:
        lines = info_file.readlines()
        features_list = list(map(lambda x: tuple(x.strip("\'").split(' = ')),
                                 lines))

        for (key, value) in features_list:
            info[key] = value.rstrip().strip("'").strip(' ')
            # if we have a number, we want it to be an integer
            if info[key].isdigit():
                info[key] = int(info[key])
    return info

###############################
# ML Freiburg General Imports
###############################

import logging
import re
import tempfile
from signal import SIGKILL, SIGTERM
import sys
import argparse

tmp = float(time.time())
timer = {'overall': tmp}

# ===== Set up logging
tmp_log_dir = tempfile.mkdtemp()
logger = logging.getLogger(name="run.py")
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] [%(asctime)s:%(name)s]: '
                           '%(message)s')

formatter = logging.Formatter(fmt='[%(levelname)s] '
                                  '[%(asctime)s:%(name)s]: %(message)s',
                              datefmt='%m-%d %H:%M:%S')
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# We don't need this
#logfile = os.path.join(tmp_log_dir, "run.log")
#file_handler = logging.FileHandler(filename=logfile, mode="w")
#file_handler.setLevel(logging.DEBUG)
#file_handler.setFormatter(formatter)
#logger.addHandler(file_handler)

# == Definitions and variable
BUFFER_BEFORE_SENDING_SIGTERM = 10  # We send SIGTERM to all processes
DELAY_TO_SIGKILL = 5  # And after a delay we send a sigkill

logger.debug("======== TIMELIMITS PER DATASET ========")
logger.debug("BUFFER_BEFORE_SENDING_SIGTERM = %s" % BUFFER_BEFORE_SENDING_SIGTERM)
logger.debug("DELAY_TO_SIGKILL = %s" % DELAY_TO_SIGKILL)
logger.debug("========================================")
###############################

# =========================== BEGIN PROGRAM ================================

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Type of the task"
    )
    parser.add_argument(
        "--train-csv",
        help="Path to train CSV file"
    )
    parser.add_argument(
        "--test-csv",
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--model-dir",
        help="Folder to save the trained model "
    )

    return parser.parse_args()


def get_parent_folder(path):
    return os.sep.join(os.path.abspath(path).split(os.sep)[:-1])

if __name__=="__main__" and debug_mode<4:
    # args = None
    args = setup_argparse()
    #### Check whether everything went well (no time exceeded)
    execution_success = True
    
    #### INPUT/OUTPUT: Get input and output directory names
    # if len(argv)==1: # Use the default input and output directories if no arguments are provided
    #     input_dir = default_input_dir
    #     output_dir = default_output_dir
    #     program_dir = default_program_dir
    # else:
    input_dir = get_parent_folder(get_parent_folder(args.train_csv))
    output_dir = os.path.join(get_parent_folder(input_dir), 'output')
    program_dir = os.path.join(get_parent_folder(input_dir), 'program')
        
    if verbose: 
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        
    # Our libraries
    path.append (program_dir + "/lib/")
    path.append (input_dir)

    import data_io                       # general purpose input/output functions
    from data_io import vprint           # print only in verbose mode
    #from models import MyAutoML    		 # example model

    ###############################
    # ML Freiburg Local Imports
    ###############################

    import util
    import logic
    from multiprocessing import Process
    import shutil
    import psutil

    ###############################

    if debug_mode >= 4: # Show library version and directory structure
        data_io.show_dir(".")
        
    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir+'_'+the_date)
    data_io.mkdir(output_dir)
    
    #### INVENTORY DATA (and sort dataset names alphabetically)
    # datanames = data_io.inventory_data(input_dir)
    # Overwrite the "natural" order
    dataname = args.train_csv.split(os.sep)[-2]
    
    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_version()
        data_io.show_io(input_dir, output_dir)
        print('\n****** Ingestion program version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')        	

        datanames = [] # Do not proceed with learning and testing

    #############################
    # ML Freiburg Code
    #############################
    # == Get overall time_limit
    overall_budget = 0
    budgets = {}

    info = get_info_from_file(input_dir, dataname)
    overall_budget += int(float(info["time_budget"]))
    budgets[dataname] = int(float(info["time_budget"])) - BUFFER_BEFORE_SENDING_SIGTERM

    #### MAIN LOOP OVER DATASETS:
    tmp = float(time.time())
    start_task = tmp
    vprint( verbose,  "\n========== ML Freiburg " + str(version) + " ==========\n")
    vprint( verbose,  "************************************************")
    vprint( verbose,  "******** Processing dataset " + dataname.capitalize() + " ********")
    vprint( verbose,  "************************************************")

    tmp = float(time.time())
    time_left = overall_budget - (tmp - overall_start)
    time_left_for_this_task = budgets[dataname] - (tmp - start_task)
    time_left_for_this_task = min(time_left_for_this_task, time_left)

    logging.info("%g sec left in total; %g sec left for %s" %
                 (time_left, time_left_for_this_task, dataname))


    print('BASENAME', dataname)
    print('input_dir', input_dir)
    print('output_dir', output_dir)

    tmp_output_dir = tempfile.mkdtemp(suffix="_" + dataname,
                                      dir=output_dir)

    p = Process(target=logic.run_automl,
                kwargs={"args": args,
                        "logger": logger,
                        "input_dir": input_dir,
                        "output_dir": output_dir,
                        "dataset_name": dataname,
                        "tmp_output_dir": tmp_output_dir,
                        "budget": time_left_for_this_task,
                        "seed": 3,
                        "sleep": 5})
    p.start()
    p.join(time_left_for_this_task)
    pid = p.pid
    if p.is_alive():
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()

    # Start shutting down
    tmp = float(time.time())
    time_left = (overall_budget - (tmp - overall_start))
    time_left_for_this_task = budgets[dataname] - (tmp - start_task)
    time_left_for_this_task = min(time_left, time_left_for_this_task)
    logger.info("%s done, %g sec left [%g sec], %g sec left in total" %
                (dataname, time_left_for_this_task, budgets[dataname],
                 time_left))

    logger.info("Starting Shutdown!")

    program_exp = re.compile(r"run\.py").search
    contacts = util.send_signal_to_our_processes(sig=SIGTERM, filter=program_exp)
    logger.debug("Sending SIG=%d to %s" % (SIGTERM, str(contacts)))

    time.sleep(DELAY_TO_SIGKILL)

    contacts = util.send_signal_to_our_processes(sig=SIGKILL, filter=program_exp)
    logger.debug("Sending SIG=%d to %s" % (SIGKILL, str(contacts)))

    logger.debug("Deleting %s" % tmp_output_dir)
    for i in range(5):
        try:
            shutil.rmtree(tmp_output_dir)
        except:
            time.sleep(1)
        if not os.path.isdir(tmp_output_dir): break


    #########################
    # ML Freiburg Code ends
    #########################
    tmp = float(time.time())
    overall_time_spent = tmp - overall_start
    if execution_success:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_budget)
    else:
        vprint( verbose,  "[-] Done, but some tasks aborted because time limit exceeded")
        vprint( verbose,  "[-] Overall time spent %5.2f sec " % overall_time_spent + " > Overall time budget %5.2f sec" % overall_budget)
