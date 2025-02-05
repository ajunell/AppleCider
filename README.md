# AppleCider: (page under construction)
## Applying multimodal learning to Classify transient Detections Early
multimodal astornomical transient classifer that uses photometry, images, spectra, and metadata. architecture based on [AstroM3](https://arxiv.org/abs/2411.08842) & [BTSbot](https://iopscience.iop.org/article/10.3847/1538-4357/ad5666)

(logo coming soon)

<i>AppleCider's name was inspired by [University of Minnesota's](https://mnhardy.umn.edu/apples) development of iconic apple cultivars. s/o Honeycrisp. </i>


# guide for guests:

How does this repo work?
- it doesn't. please come back later. <br>
- there's also a seperate data-centered repo, [AppleCider Data](https://github.com/ajunell/AppleCider_Data) (also private), which is actually organized.
  - it contains:
    - notebooks with stats about the dataset (class distribution, instrument counts, host galaxy spectra info, etc)
    - all the queries (sans tokens) I used: Kowalski for alerts, Fritz for classification + spectra + additional object information, SDSS for more spectra, and DESI also for more spectra. there are also example notebooks for querying and basic spectra processing (which is really just saving each file format to .csv). note: there is additioanl spectra I got from Yu-Jing, so no query notebooks for that, just processing.
    - relevant .csv: the classic object id + classifcation, object id + additional info (RA, Dec, named host galaxy, spectra source + spectral classification + spec ids for each survey, etc), object id + host galaxy name + potential SDSS host galaxy spec ID + SDSS file name 
