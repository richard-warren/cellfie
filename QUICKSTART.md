# quick start

- initial setup
    - clone this repo and add the directory containing it to PYTHONPATH
    - make sure the following are installed: numpy, scipy, cv2, tifffile, pillow, pyyaml, tqdm, tensorflow-gp (requires version 2.2, currently only available with pip install), pandas, skimage, matplotlib
    - create a data directory containing the folders
        - *datasets:* contains one folder for each calcium imaging dataset, labelled 'images_J115', 'images_J123', ...
        - *labels:* contains the manual labels (i can provide this)
        - *models:* contains an *instance_segmentation* and *region_proposal* folder. models will be generated and stored here.
        - *results:* generated plots will be stored here.
        - *training_data:* summary images will be generated and stored here.
    - edit `config.yaml` to point to the `data_dir` created above, and to list all of the datasets you want to use.

- make summary images
    - adjust settings in `prepare_training_data/config.yaml`
    - run ` python prepare_training_data/prepare_training_data.py`
    - numpy files containing training data, and .png files for the summary images will appear in `data_dir/training_data`
    - this may take a while. mean, median, max, std, and correlation images are all being computed. you only have to do this once though!
    
- region proposal network training
    - adjust settings in `region_proposal/config.yaml`, then run `region_proposal/train.py`
    - models will appear in `data_dir/models/region_proposal/DATETIME`. sample images will also be generated to show the input data (top row), ground truth (middle row), and output (last row) of the current model evaluated on the test data.
    - after training is complete, all but the best model (as determined by test set accuracy) will be deleted.

- instance segmentation network training
    - adjust settings in `instance_segmentation/config.yaml`, then run `region_proposal/train.py`
    - models will appear in `data_dir/models/instance_segmentation/DATETIME`. sample images will also be generated to show the input data (top row), ground truth (middle row), and output (last row) of the current model evaluated on the test data.
    - after training is complete, all but the best model (as determined by test set accuracy) will be deleted.

- putting it all together
    - `end_to_end/analyze_all_vids.py` is a template for analyzing all of the videos "end-to-end". it loads the previously computed summary images, finds potential neurons by looking for maxima in the region_proposal network output, and then segments/classifies potential neurons by running subframes centered at these locations through the instance_segmentation network.
    - Choose the models you want by adjusting `rp_model` and `is_model`. They should point to the region proposal and instance segmentation models you want to use, which can be found in `data_dir/models`. 
    - Adjust as necessary `maxima_thresh` and `min_distance`, which affects which region proposal maxima are passed along to the instance segmentation network (see in-line comments).
    - Running the script will analyze all datasets listed in `cellfie/config.yaml`. It will save figures summarizing the results to `data_dir/results/figures`, and save the output as .npz files in `data_dir/results/masks`. The red dots on the image in the middle shows where maxima were found in the output of the region proposal network. Use this to adjust the 'maxima_thresh' parameter.     
