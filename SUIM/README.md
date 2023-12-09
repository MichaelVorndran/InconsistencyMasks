# SUIM


## Step 1: Download the Dataset

Go to https://drive.google.com/drive/folders/10KMK0rNB43V2g30NcA1RYipL535DuZ-h and download "train_val.zip" and "TEST.zip".


![download image](download_suim.png)


## Step 2: Specify the Base Directory and Extract the Downloaded Files
First, ensure that you've specified your base directory for this dataset in your `config.ini` file. For example:

    BASE_DIR = C:/IM/SUIM/

Inside the base directory you've specified, create a folder named `original_data`. Using the example above, this would be:

    C:/IM/SUIM/original_data/

Once the files are downloaded, extract the data into the `original_data` folder. This can be done using built-in OS tools or third-party tools like 7-Zip or WinRAR.

After extracting the files, the contents of the `original_data` folder should look like this:

![original_dir_image](original_data_suim.PNG)


## Step 3: Create the Datasets and Start Training

Next, execute all Python scripts in the provided order to reproduce all results. 

For those who wish to test specific approaches, you must at least generate the training data using the scripts 
`00_SUIM_convert_bmp_to_png_masks.py` and `01_SUIM_split_original_train_val.py`, and train the subset models using `03_SUIM_subset.py` or `04_SUIM_subset_aug.py`.


## Reference
M. J. Islam u. a., „Semantic Segmentation of Underwater Imagery: Dataset and Benchmark“, 2020, doi: 10.48550/ARXIV.2004.01241.

https://github.com/xahidbuffon/SUIM
