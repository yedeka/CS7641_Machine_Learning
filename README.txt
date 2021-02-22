CS7641 - Machine Learning 
Yogesh Edekar (GTID - yedekar3)

Environment - Anaconda, Pycharm (IDE). 
Source Code git repo - https://github.com/yedeka/CS7641_Machine_Learning
Data - Available in the same git repo

environment setup - 
1] To setup Conda environment please download the source code and locate environment.yml file in the root folder.
2] If anaconda is not already installed on the system please install anaconda. 
3] On conda prompt please execute following commands for setting up the environment. 
	conda env create --file <Path_to_environment.yml>
	conda activate ml
	pip install yellowbrick
4] If any issue is faced with imbalanced-learn library it can be removed. It is an extra library that was kept initially for handling imbalanced datasets.
5] The entire code has been developed using pycharm. 
6] Please install pycharm. 
7] Create a new project by importing the source code from git repo.
8] Selected existing pythin interpreter from the conda environment to be used within pycharm so that all the dependencies for the project will be automatically made available.
9] run the testcode.py file to run both the experiments and obtain the charts as well as the observations for the experiments. 		