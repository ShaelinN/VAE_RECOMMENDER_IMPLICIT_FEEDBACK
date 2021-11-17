# VAE_RECOMMENDER_IMPLICIT_FEEDBACK
VAE Collaborative Filtering Recommender System using implicit feedback from Yelp dataset 

# HOW-TO:
## -1: CLARIFICATIONS
a more legible version of this file is proved as HOW-TO.md
The project assumes you will be using Google Colab for the most part to run code 
The project assumes you will be operating out of a Google Drive folder for the most part. make sure to mount the google drive in the runtimes used, for each notebook. this may or may not require the opening of a new window to confirm authorisation (follow whatever prompts are given).

=======================================================================================================================================================================================================================================================================================================================================================================================

## 0:DESIGNATE A ROOT FOLDER FOR THE PROJECT
make a directory to house the various workings of the project. 
=======================================================================================================================================================================================================================================================================================================================================================================================


##1: GETTING THE ORIGINAL DATA
the yelp dataset is linked elsewhere in this submission. once it is downloaded, it is necessary to extract the relevant information out of the relevant files. Due to the size of these files and the fact that my system would not allow me to access such a large file, I had it broken into chunks by line, uploaded to google drive, and then reassembled into a single csv file.

I have provided the link to this csv in the same file as the link to the original dataset, to bootstrap the process for whomever it may concern.

I have also provided links to my local copy of the MovieLens reviews in the same format as my copy of the yelp data so that the project can run on either of them with minimal issues

Make a directory called "Data" inside the root directory, and place th reviews.csv file (as well as the ml10m_reviews.csv file if you intend to work with MovieLens too) inside it
=======================================================================================================================================================================================================================================================================================================================================================================================


## 2: PREPROCESSING
open the notebook Preprocessing.ipynb
under the section "Common directories", change the value of the variable called  root  too much the root folder of the project on your system.  

change the value of the variable called specific_version_data  to something that the dataset you are working with,  example "yelp", or "ml10m".

 change the name of the variable records_filename to match the name of the reviews file that you want to read from.  for the yelp dataset this file is called "reviews.csv" , and for movielens 10M it is called "ml10m_reviews.csv"

Do not change any of the other directories. 

Run all code in the "All imports" section.
Run all code in  the "Common directories" section

the rest of the sections may be run independently if there was a crash, provided that the imports and directories are loaded.

If you are dealing with MovieLens dataset,  the uid and bid columns have numeric values that may throw off some of the operations that will be done later. To overcome this run the section called "Hack", Which will replace the original data file with the new version that is forced to use string values for IDs. If you are dealing with yelp data you do not need to run the section called "Hack"  since the IDs are already in a string format.

Within the section called "Filtration" change the value of the variables "user_threshold" and "business_threshold". These represent the minimum amount of interactions that a user or business must have in order to be carried through to the filtered data set. 

Run everything in both subsections within the section called "Filtration". This will filter pandas dataframe of records and then save the filtered results as CSV.  it will also save a randomly ordered list of all users and businesses in the filtered data set.
 Run all code in the section called "Binarize Data Format". It will save its results.

 Run all code in all subsections of  the section called "Split". It will save its result as individual split csvs and some text files

Run all code in all subsections of the section called "To Matrix". This will save a series of pickle files. However, in all subsequent work the explicit matrices have not been used, and realistically you may delete them, keeping only the implicit matrices. DO NOT delete any text files or any csv files, however.
=======================================================================================================================================================================================================================================================================================================================================================================================

## 3:VAE
Open notebook VAE.ipynb. Ensure that you are using a GPU-runtime

some Pip install commands are provided in the first cell for libraries that MAY require installation before they are ready to work in Google colab. If there is an error thrown because of a failed import, uncomment the relevant install command or add a new install command to the first cell, and run that cell.

Imports and Dirs:
There is a large commented-out block involving the library argparse. If you are running this code on Google colab or any kind of Jupyter notebook it is not necessary to touch this 	comment,  as the  next block of code defines a class that serves as an analogue to this code. 
Within the class called argclass, change the value of the variable "self.root" to match the root directory of your project. 
Change the value of "self.input_data"  such that it points to the same location as "specific_version_data" (not the main "Data" directory) that you defined in the "Preprocessing" notebook.
Give "self.training_results" a value which adequately describes the outputs of this file.  the default value of "vae_ml10m_training_results"  for example means that we are training a vae on the ML-10M dataset. 
If you are  rerunning this code after the primary training phase crashed before completing all its epochs, then "self.weights" will be used to load any model saved before the crash
If you would like you may change the value of "self.intermediatedim", "self.epochs", "self.batchsize" and "self.klannealrate"
Run all code in the section "Imports and Dirs". Ensure that it has indeed initialized your training results directory, within the root folder of the project. 

Model Design
Run all code within the "Model Design" section. It provides a means to build vae, as well as the loss function and a data generator

NOTE:All sections below (still in the notebook VAE.ipynb) requires you have run the "Imports and Dirs" Section. "TRAINING" and "Evaluation" also require you have run "Model Design" section. "TRAINING", "Analysis" and "Evaluation" do not require each other’s variables, provided the all file writes (in TRAINING) went through without issues.

TRAINING
Run all code within the "TRAINING" section. This will take a long time. The "Primary Train" subsection will take up most of the time. 

Analysis 
If you want to see the loss curves of the data,run  the first block of code under Analysis. Change the variable "relevant_loss_data" to match the data you want to see. The available loss files are outlined in argclass, with explanation of what it is commented next to it.
If you want to see some predictions, run the second block of code under analysis.Change the value of item_num and batch_num  to predict on different users. 

Evaluation 
Run all the cells in the Evaluation section. This will produce the recalls and ndcg at k for the VAE

