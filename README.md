# VAE_RECOMMENDER_IMPLICIT_FEEDBACK
VAE Collaborative Filtering Recommender System using implicit feedback from Yelp (and MovieLens) dataset(s) 

# HOW-TO:
# LIBRARY VERSIONS:
* Python: 3.7/3.9
* Tensorflow: 2.7.0/2.6.0 
* Numpy: 1.19.5
* Pandas: 1.1.5
* Surprise: 1.1.1/0.0.5
* Pickle: 4.0

# CLARIFICATIONS
* The project assumes you will be using Google Colab for the most part to run code 
* The project assumes you will be operating out of a Google Drive folder for the most part. make sure to mount the google drive in the runtimes used, for each notebook. This may or may not require the opening of a new window to confirm authorisation (follow whatever prompts are given).
* it is possible to convert the models' .ipynb files to regular .py files to run on a cluster such as HIPPO at the University of KwaZulu-Natal, however you must ensure that all libraries are installed, that data folders moved to the cluster are not modified outside of code(i.e. the data folders called specific_version_data in Preprocessing, not the main folder called "Data"). Also ensure that any Colab-specific code is replaced with the relevant analogues (e.g. replacing argclass with the implementation of argparse parser), and that you call the .py file with the correct arguments. Naturally **it is easier to just run in Colab if the aim is demonstration**.


Each section describes a different way to make use of this project, starting from the simplest way in Section A, to the most complicated in Section C.

# DESIGNATE A ROOT FOLDER FOR THE PROJECT
make a directory to house the various workings of the project. For accessing any of the content currently on my Google drive, you may create a shortcut to the file/s in your Google Drive, within this root folder. If you mount your Google Drive in Colab, it will be able to seamlessly access the shortcuts as well.

# SECTION A: BOOTSTRAP FROM PRE-TRAINED MODEL:
This section covers how to access my pretrained models so you don't need to do data preprocessing or model training yourself.
Section B covers training a model by bootstrapping from an already prepared dataset. 
Section C explains how to go from a reviews file to fully trained models.
## A1: GETTING MODELS:
* these links will get you the trained VAE and SVD++ models:
>VAE/Yelp: https://drive.google.com/file/d/1lasGCIvW-g8aW4ntXhqE8_2aVFE9MgHa/view?usp=sharing 

>VAE/ML-10M: https://drive.google.com/file/d/1-ke_i9z7tfRmPT2fjmJgCUM_dx9_KVm7/view?usp=sharing

>SVD++/Yelp: https://drive.google.com/file/d/1-2GM6Qswc0gniuqYy9m_HuXJQ0ocWHiB/view?usp=sharing

>SVD++/ML-10M: https://drive.google.com/file/d/1ooJ02J9ciKbjacof4wYbatIJLAuWeIlW/view?usp=sharing
* place each of these models in their own directory within the project root folder

## A2: GETTING DATASET:
* these links lead to the datasets needed for the models to be evaluated:
> Yelp: https://drive.google.com/drive/folders/1DJ57HdrrJd8yMqIrJ8MKxFmTxAefBmt2?usp=sharing
 
> ML-10M: https://drive.google.com/drive/folders/1XnWpFcGKvrkCTUNFC4Z8HcTSTnM6E6HS?usp=sharing
* unzip and place the datasets' folders somewhere within the project root folder.

## A3: VARIABLES:
for VAE use the VAE.ipynb notebook, for the SVD++ baseline, go to SVD++.ipynb
* set the "self.root" value (under the class "argclass" in the imports and dirs section) to be the absolute path of the project root directory.
* set the "self.training_results" value (under the class "argclass" in the imports and dirs section) to be the relative path of this directory compared to the project root directory.
* set the "self.input_data" value (under the class "argclass" in the imports and dirs section) to be the relative path of this directory compared to the project root directory.

## A4: EVALUATION/USAGE:
* run "Imports and Dirs" sections in the notebook of the model being used. This will store the directories for the data and model. 
* you may need to install some libraries using pip or to mount google drive.
* if you are evaluating the SVD++ baseline, run the predictions section to generate a prediction matrix.
* if you are evaluating the actual VAE, run the Model Design section to create the infrastructure to build the model and data generator
* for either model type, run "Evaluation" section in the notebook

# SECTION B: BOOTSTRAP FROM PREPROCESSED DATA
This section explains training a model by bootstrapping from an already prepared/preprocessed dataset. 
Section A covers bootstrapping from an already trained model. 
Section C explains how to go from a reviews file to fully trained models.

## B1: GETTING DATASET:
* follow the instructions in Section A2

## B2: TRAINING, EVALUATION:
* follow the instructions in Section C3 (or C4 for SVD++). 
  * Note that the guide mentions a value/variable "specific_version_data" from the Preprocessing stage.
  * This refers to the folder of your preprocessed dataset if it was generated in Section C2. 
  * If you are following Section B and downloaded the preprocessed dataset, then consider "specific_version_data" to refer to the path of this downloaded dataset relative to the project root
  * as such, set args.input_data to point to the folder of the downloaded dataset.

    

# SECTION C: FULL RUN
this section explains how to go from a reviews file to fully trained models. 
Section A covers bootstrapping from an already trained model. 
Section B covers training a model by bootstrapping from an already prepared dataset.
Preprocessing notebook must be run completely before running VAE notebook or SVD++ notebook

## C1: GETTING THE ORIGINAL DATA
* Here is the link to csv files of review tuples. 
    > Yelp: https://drive.google.com/file/d/1cdWbyWTNc7FtP03MQr5m0KN_J2omBue9/view?usp=sharing

    > MovieLens 10M: https://drive.google.com/file/d/1RwmlMUSNhsv4eitt6M7lN0N6sYp-MlIx/view?usp=sharing

  * For the rest of this tutorial, it is assumed you are working with these file and not the original datasets from the relevant websites (yelp and grouplens)
  * These files do originate in the original dataset but have undergone some cleaning and standardisation

* The original datasets can be found at:
    > Yelp: https://www.yelp.com/dataset

    > MovieLens: https://grouplens.org/datasets/movielens/10m/

    * These original datasets have way more information than the required fields and some cleaning occured before preprocessing to be able to move then easily between machines. 
    * I advise the use of the review tuple CSV files linked in the point above this, as they contain only the usable data and are of standardised format

Make a directory called "Data" inside the root directory, and place the reviews.csv file (as well as the ml10m_reviews.csv file if you intend to work with MovieLens too) inside it

## C2: PREPROCESSING
open the notebook Preprocessing.ipynb

### All imports
* Run all code in the "All imports" section.

### Common directories
* change the value of the variable called  root  too match the root folder of the project on your system.  
* change the value of the variable called specific_version_data  to something that represents the dataset you are working with, example "yelp", or "ml10m".
* change the name of the variable records_filename to match the name of the reviews file that you want to read from. Using the provided CSVs, the yelp dataset file is called "reviews.csv" , and for movielens 10M it is called "ml10m_reviews.csv"
* DO NOT change any of the other directories. 
* Run all code in  the "Common directories" section

### the rest of the sections may be run independently if there was a crash, provided that the imports and directories are loaded.

### HACK
If you are dealing with MovieLens dataset,  the uid and bid columns have numeric values that may throw off some of the operations that will be done later. To overcome this run all code in the section called "Hack", which will replace the original data file with the new version that is forced to use string values for IDs. If you are dealing with yelp data you do not need to run the section called "Hack"  since the IDs are already in a string format.

### Filtration
Within the section called "Filtration" change the value of the variables "user_threshold" and "business_threshold". These represent the minimum amount of interactions that a user or business must have in order to be carried through to the filtered data set. 

* Run everything in both subsections within the section called "Filtration". This will filter pandas dataframe of records and then save the filtered results as CSV.  it will also save a randomly ordered list of all users and businesses in the filtered data set.

### Binarize Data Format
Run all code in the section called "Binarize Data Format". It will save its results.

### Split
Run all code in all subsections of  the section called "Split". It will save its result as individual split csvs and some text files

### To Matrix
Run all code in all subsections of the section called "To Matrix". This will save a series of pickle files. However, in all subsequent work the explicit matrices have not been used, and realistically you may delete them, keeping only the implicit matrices. DO NOT delete any text files or any csv files, however.


## C3: VAE
Open notebook VAE.ipynb. Ensure that you are using a GPU-runtime

some Pip install commands are provided in the first cell for libraries that MAY require installation before they are ready to work in Google colab. If there is an error thrown because of a failed import, uncomment the relevant install command or add a new install command to the first cell, and run that cell.

### Imports and Dirs:
There is a large commented-out block involving the library argparse. If you are running this code on Google colab or any kind of Jupyter notebook it is not necessary to touch this 	comment,  as the  next block of code defines a class that serves as an analogue to this code. 
Within the class called argclass: 
* change the value of the variable "self.root" to match the root directory of your project. 
* Change the value of "self.input_data"  such that it points to the same location as "specific_version_data" (not the main "Data" directory) that you defined in the "Preprocessing" notebook.
* Give "self.training_results" a value which adequately describes the outputs of this file.  the default value of "vae_ml10m_training_results"  for example means that we are training a vae on the ML-10M dataset. 
* If you would like you may change the value of "self.intermediatedim", "self.epochs", "self.batchsize" and "self.klannealrate"
* Run all code in the section "Imports and Dirs". Ensure that it has indeed initialized your training results directory, within the root folder of the project. 

### Model Design
Run all code within the "Model Design" section. It provides a means to build vae, as well as the loss function and a data generator

### NOTE:All sections below (still in the notebook VAE.ipynb) requires you have run the "Imports and Dirs" and "Model Design" sections. "TRAINING", "Analysis" and "Evaluation" do not require each other’s variables, provided the all file writing (in TRAINING) went through without issues.

### TRAINING
Run all code within the "TRAINING" section. This will take a long time. The "Primary Train" subsection will take up most of the time. 

### Analysis 
* If you want to see the loss curves of the data,run  the all code in the first subsection under Analysis. 
* If you want to see some predictions, run the code in the second section under Analysis. Change the value of u_num to predict on different users. 

### Evaluation 
Run all the cells in the Evaluation section. This will produce the recalls and ndcg at k for the VAE

## C4: SVD++/Baseline

### Imports and Dirs
similar to the VAE there is a commented out block using argparse and an analogue for it called argclass. within  this class:
* change the value of the variable called "self.root"  too match the root folder of the project on your system.  
* Change the value of "self.input_data"  such that it points to the same location as "specific_version_data" (not the main "Data" directory) that you defined in the "Preprocessing" notebook.
* Give "self.training_results" a value which adequately describes the outputs of this file.  the default value of "svd_ml10m_training_results"  for example means that we are training svd(++) on the ML-10M dataset. 
* Run all code in  the "imports and dirs" section

### NOTE: All further sections in this notebook (TRAINING, Predictions and Evaluation) depend on Imports and Dirs section, but do not require each other's variables, provided all file writing has occured with no errors.

### TRAINING
Run all code in this section. This will prepare the review tuples from the csv files into an implementation-specific "trainset" needed for SVD++ to train, then train the model and save it. This will take a long time

### Predictions
Run all code in this section. this generates a matrix of prediction events for svd++ that can pe evaluated against the testing data, and saves it. You may delete the .csv temp file, but do not delete the .pkl file.

### Evaluation
Run all the code in the Evaluation section. This will produce the recalls and ndcg at k for the SVD++ model
