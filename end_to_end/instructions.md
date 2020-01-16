# getting started

* Clone the `cellfie` repo and add it to PYTHONPATH so it can be imported. All scripts should be run from the `cellfie` root directory.
* Download the `cellfie_data` folder I sent, which contains models, training data, etc. Results will be written in this folder by default.
* Modify `cellfie/config.yaml`:
    * Change `data_dir` to reflect the location of the `cellfie_data` folder.
    * Change `datasets` to include only the datasets you would like to analyze.
* Anaylze data:
    * `end_to_end/analyze_all_vids` is a template for analyzing all videos.
    * Choose the models you want by adjusting `rp_model` and `is_model`. They should point to the region proposal and instance segmentation models you want to use, which can be found in `cellfie_data/models`. 
    * Adjust as necessary `maxima_thresh` and `min_distance`, which affects which region proposal maxima are passed along to the instance segmentation network (see in-line comments).
    * Running the script will analyze all datasets listed in `cellfie/config.yaml`. It will save figures summarizing the results to `cellfie_data/results/figures`, and save the output as .npz files in `cellfie_data/results/masks`