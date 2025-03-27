<h1> :green_apple: <code style="color : grey"> AppleCiDEr </code> :green_apple: <br> </h1> 
<h4> <code style="color : grey">APPlying multimodaL lEarning to Classify transIent Detections EaRly </code></h4>

`AppleCiDEr` is an (in progress) multimodal transient classifer that uses photometry, images, spectra, and metadata. architecture based on [AstroM3](https://arxiv.org/abs/2411.08842) & [BTSbot](https://iopscience.iop.org/article/10.3847/1538-4357/ad5666). <sup><i>(logo coming in the near-ish future)</i></sup>

<br>

> <i>`AppleCiDEr`'s name was inspired by [University of Minnesota's](https://mnhardy.umn.edu/apples) development of iconic apple cultivars. s/o Honeycrisp. </i><br>

the public version of `AppleCiDEr` (this repo) makes use of objects from the [ZTF Bright Transient Survey](https://sites.astro.caltech.edu/ztf/bts/bts.php).<br>

## guide for guests:
$${\color{red}note:}$$ $${\color{red}this}$$ $${\color{red}repo}$$ $${\color{red}is}$$ $${\color{red}under}$$ $${\color{red}construction}$$ $${\color{red}as}$$ $${\color{red}of}$$ $${\color{red}3/27}$$. see `intro_notebook.ipynb` for a general overview of things. right now, the contents of `/notebooks` is out of date and only for viewing purposes.

<ins>What do I really need to download from this repo?</ins>
- everything in the`AppleCider` folder (core, models, preprocess)
- plus files in `csv-pkl`:
  - `SEDM_BrightTransientSurvey.csv` <br>
  - depending on if you want to truly re-do the data preprocessing steps (outlined in `001-data-processing.ipynb`) or jump ahead using files already in the repo, you will need some version of:
    - `data_train_BTS.csv`, `data_test_BTS.csv`
<br>

<ins>How to drink Apple Cider</ins>:
- `drink-AppleCider.ipynb`: bare bones notebook to run the model, contains three cells: imports, config, function to run the model. requires processed data (see `001-data-processing.ipynb` for formatting of data)
- `/notebooks/006-AppleCider-multimodal`: walk through AppleCider. includes printed examples of processed data, example use of `DataGenerator` + `DataLoader`, graph original photometry vs processed photometry, mass graph images + photometry + spectra and print metadata columns for alerts in `DataGenerator`. 
  - if you want to look at the individual modalities, see the other notebooks in the folder




