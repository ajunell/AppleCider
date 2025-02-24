# AppleCider: (page under construction)
## Applying multimodal learning to Classify transient Detections Early
multimodal transient classifer that uses photometry, images, spectra, and metadata. architecture based on [AstroM3](https://arxiv.org/abs/2411.08842) & [BTSbot](https://iopscience.iop.org/article/10.3847/1538-4357/ad5666)

(logo coming soon)

<i>AppleCider's name was inspired by [University of Minnesota's](https://mnhardy.umn.edu/apples) development of iconic apple cultivars. s/o Honeycrisp. </i>


# guide for guests:

Give me the data:
- [AppleCider Data](https://drive.google.com/drive/folders/13x2KGVOHwkO_VmrcNHnEnuQgT0yT_BL_?usp=sharing)
  - folder formated basically same as on Theophile's machine: each object has it's own folder which contains photometry.csv, alerts.npy, spectra.csv
  - you can use `SEDM_folder` the same as it's used in the repo (just change paths to wherever `SEDM_folder` is on your machine)
 
  <br>

How does this repo work?
- `000-drink-AppleCider.ipynb` bare bones way to run model (no graphs, only has config & line to run the model). although, it does require preprocessed data....
- `AppleCider/preprocess`: for preprocessing objects in dataset into multiple alerts. see [notebooks/001-data-processing.ipynb](https://github.com/ajunell/AppleCider/blob/main/notebooks/001-data-processing.ipynb) for how to use the preprocessing steps / what the processed alerts look like.
- `/AppleCider/core`: for model stuff, see [002-AppleCider-photometry.ipynb](https://github.com/ajunell/AppleCider/blob/main/notebooks/002-AppleCider-photometry.ipynb), [notebooks/003-AppleCider-metadata.ipynb](https://github.com/ajunell/AppleCider/blob/main/notebooks/003-AppleCider-metadata.ipynb), [notebooks/005-data-processing.ipynb](https://github.com/ajunell/AppleCider/blob/main/notebooks/001-data-processing.ipynb)





<br><br><br><br>
- <s> it doesn't. please come back later. <i>after i resolve another RuntimeError in `002-run-model.ipynb`(not the one currently there, i fixed that one), i'll write up an explanation. </i> <br>  for photometry, see `2-19_notebook.ipynb` </s>

- there's also a seperate data-centered repo, [AppleCider Data](https://github.com/ajunell/AppleCider_Data) (also private), which is actually organized.
  - it contains:
    - notebooks with stats about the dataset (class distribution, instrument counts, host galaxy spectra info, etc)
    - all the queries I used: Kowalski for alerts, Fritz for classification + spectra + additional object information, SDSS for more spectra, and DESI also for more spectra. there are also example notebooks for querying and basic spectra processing (which is really just saving each file format to .csv). note: there is additional spectra I got from Yu-Jing, so no query notebooks for that, just processing.
    - relevant .csv: the classic object id + classifcation, object id + additional info (RA, Dec, named host galaxy, spectra source + spectral classification + spec ids for each survey, etc), object id + host galaxy name + potential SDSS host galaxy spec ID + SDSS file name
  - it does NOT contain data... see AppleCider drive link above
