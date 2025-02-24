# AppleCider:
AppleCider (applying multimodal Learning to classify transient detections early): a multimodal transient classifer that uses photometry, images, spectra, and metadata. architecture based on [AstroM3](https://arxiv.org/abs/2411.08842) & [BTSbot](https://iopscience.iop.org/article/10.3847/1538-4357/ad5666)

(logo coming soon)

<i>AppleCider's name was inspired by [University of Minnesota's](https://mnhardy.umn.edu/apples) development of iconic apple cultivars. s/o Honeycrisp. </i>


## guide for guests:

<ins>Give me the data:</ins>
- [AppleCider Data](https://drive.google.com/drive/folders/13x2KGVOHwkO_VmrcNHnEnuQgT0yT_BL_?usp=sharing): `SEDM_folder` is formated basically same as on Theophile's machine: each object has it's own folder which contains `photometry.csv`, `alerts.npy`, `spectra.csv`
- this is NOT the preprocessed data
<br> 


<ins>What do I really need to download from this repo?</ins>
- everything in the`AppleCider` folder (core, models, preprocess)
- plus the loose file(s) in the repo:
  - `SEDM_dataset.csv` <br>
  - depending on if you want to truly re-do the data preprocessing steps (outlined in `001-data-processing.ipynb`) or jump ahead using files already in the repo, you will need some version of:
    - `data_train.csv`, `data_test.csv`,
    - `test_files.pkl`, `train_files.pkl`, `val_files.pkl`
<br>

<ins>How to drink Apple Cider</ins>:
- `drink-AppleCider.ipynb`: bare bones notebook to run the model, contains three cells: imports, config, function to run the model. requires processed data (see `001-data-processing.ipynb` for formatting of data)
- `/notebooks/006-AppleCider-multimodal`: walk through AppleCider. includes printed examples of processed data, example use of `DataGenerator` + `DataLoader`, graph original photometry vs processed photometry, mass graph images + photometry + spectra and print metadata columns for alerts in `DataGenerator`. 
  - if you want to look at the individual modalities, see the other notebooks in the folder 




<br><br><br>

### AppleCider Data Repo

- there's also a seperate data-centered repo, [AppleCider Data](https://github.com/ajunell/AppleCider_Data) (also private), which is actually organized. it contains:
  - notebooks with stats about the dataset (class distribution, instrument counts, host galaxy spectra info, etc)
  - all the queries I used: Kowalski for alerts, Fritz for classification + spectra + additional object information, SDSS for more spectra, and DESI also for more spectra. there are also example notebooks for querying and basic spectra processing (which is really just saving each file format to .csv). note: there is additional spectra I got from Yu-Jing, so no query notebooks for that, just processing.
  - relevant .csv: the classic object id + classifcation, object id + additional info (RA, Dec, named host galaxy, spectra source + spectral classification + spec ids for each survey, etc), object id + host galaxy name + potential SDSS host galaxy spec ID + SDSS file name
 - it does NOT contain data... see AppleCider drive link above
