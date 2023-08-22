# lightcurve_extraction
Script to reduce, plate solve and perform photometry on raw astronomical images. The script extracts a lightcurve for a given moving object and filter band. This code can also be used to obtain colours of the object based on multi filter observations. 

## Requirements
calviacat (Kelley & Lister. 2019)
photutils
astroquery

## File Structure
You must first create a folder for one observing run. Within this, there should be a folder called 'flats' containing all good flat field images for the run (providing this does not spread over too many days). There should also be folders for each nights' bias files named with 'Bias' and the day at beginning of observation - e.g. 'Bias16' for the bias images on the 16th. Each new telescope object pointing should then have a folder with its name - e.g. 'Ceres'. Within this, there should be a folder for each night of observing named in the format 'YYYYMMDD'. Once in this folder, the raw unzipped science images should be placed in a folder named 'Data'. 

The input path should be given as the path to the raw science data for that night.

Example of input path:

./VLT_March_2023/Ceres/20230312/Data

Example of bias path:

./VLT_March_2023/Bias12/r987654.fit

Example of flats path:

./VLT_March_2023/flats/r13579.fit

N.B. the bias and flat image paths can be set manually using the 'bias_path' and 'flat_path' class attributes.

## Initial Parameters
name : str

name of moving object
    
input_path : str

directory for raw science images
    
telescope_id : int

id number for telescope location in JPL Horizons system

## Outputs
#### All outputs are saved in the input path directory.

Lightcurves - Name format 'day_objectname_filter_lightcurve.png'

FWHM - Dataframe of fwhm determined by aperture photometry with name format 'filter_fwhm_list.csv'

Reference frame index - Dataframe containing the index of reference frame for the filter observations. Name format is 'filter_ref_frame_index.csv'

Results - Dataframe of all magnitude results containing Time, calibrated magnitude, magnitude error and filter band. Name format is 'objectname_results.csv'

Colours - Dataframe of all the colours determined from multi filter observations. Name format is 'objectname_colour.csv'

## Methods
- extract_and_overscan: extract only specified hdu extension and do overscan correction

- master_bias: use a median combine to create a master bias

- master_flats: use a median combine to create a master flat for each filter

- science_reduction: do all bias and flat correction on science images

- plate_solve: solve field for each image and save wcs header

- iterative: run iterative psf fitting to get an accurate stellar fwhm for each image

- detect_and_phot: run source detection and aperture photometry on each image using psf fitted fwhm

- ephemerides: query JPL Horizons system for ephemerides of object

- catalogue_stars: match source detection to PanSTARRS1 catalogue and extract only stars in all images

- calibration: perform magnitude calibration with colour correction to frame with best fwhm

- magnitudes: calculate relative magnitudes for all images and scale to get final magnitudes

- save_results: Save results dataframe as csv file

- get_fwhm: Get fwhm from csv file

- search_field: search a specified image field for moving object

- plot_lightcurve: plot and return calibrated lightcurve

- full_data_reduction: complete all data reduction steps in one

- get_epoch_range: obtain start and end epochs of science data set

- get_filters: obtain filters list

- get_ref_index: obtain refernce frame index

- extract_lightcurve: carry out magnitude calculation and calibration and plot lightcurve for each filter

- extract_single_lightcurve: carry out magnitude calculation and calibration and plot lightcurve for one filter

- colour_estimate: obtain colours of object from multifilter observation


## Acknowledgements and References
This code was developed through funding from the University of Edinburgh School of Physics and Astronomy (Vacation Scholarship).

Kelley, Michael S. P. & Lister, Tim. 2019. mkelley/calviacat. DOI: 10.5281/zenodo.2635840

