

<u>sooooooo, what's going on here</u>
- `001-data-processing.ipynb`: process the data from `SEDM_folder` (see google drive link on main page of repo)

- as for the rest, they require running `001-data-processing.ipynb` to get the processed dataset. you can skip to the `TransientDataset` portion of the notebook and use the obj_id, type columns from `data_test.csv`, `data_train.csv` in the place of `test_df`, `train_df` created with `DataSorter` to generate the processed data. 
  - `002-AppleCider-photometry.ipynb`: just photometry version
  - `003-AppleCider-metadata.ipynb`: metadata 
  - `005-AppleCider-multimodal.ipynb`: model with all modalities
