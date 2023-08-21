#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:00:00 2023

author: Isaac Holst
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from pathlib import Path
import ccdproc as ccdp
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.io import fits
from astroquery.jplhorizons import Horizons
from astropy.table import Table
import subprocess
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.psf import (IntegratedGaussianPRF, DAOGroup, IterativelySubtractedPSFPhotometry)
from astropy.modeling.fitting import LevMarLSQFitter
import calviacat as cvc
from astropy.stats import SigmaClip
from photutils.aperture import (ApertureStats, CircularAnnulus, CircularAperture)
from astroquery.imcce import Skybot
import operator as op
import numpy.ma as ma
from scipy import odr
from photutils.background import MMMBackground



class lightcurve(object):
    """
    Class to do astronomic photometry on a given set of images
    
    Attributes
    ----------
    name: str
        name of the moving object imaged
    input_path: str
        directory for science images
    bias_path: str
        directory for bias images. Set automatically unless otherwise set
    flat_path: str
        directory for flat images. Set automatically unless otherwise set
    science_basenames: list
        list of science image names
    bias_basenames: list
        list of bias image names
    flat_basenames: list
        list of flat image names
    n: int
        extension number for hdu
    telescope_id: int
        id number for telescope location in JPL Horizons system
    
    Methods
    -------
    __init__
    __str__
    extract_and_overscan: extract only specified hdu extension and do overscan correction
    master_bias: use a median combine to create a master bias
    master_flats: use a median combine to create a master flat for each filter
    science_reduction: do all bias and flat correction on science images
    plate_solve: solve field for each image and save wcs header
    iterative: run iterative psf fitting to get an accurate stellar fwhm for each image
    detect_and_phot: run source detection and aperture photometry on each image using psf fitted fwhm
    ephemerides: query JPL Horizons system for ephemerides of object
    catalogue_stars: match source detection to PanSTARRS1 catalogue and extract only stars in all images
    calibration: perform magnitude calibration with colour correction to frame with best fwhm
    magnitudes: calculate relative magnitudes for all images and scale to get final magnitudes
    save_results: Save results dataframe as csv file
    get_fwhm: Get fwhm from csv file
    search_field: search a specified image field for moving object
    plot_lightcurve: plot and return calibrated lightcurve
    full_data_reduction: complete all data reduction steps in one
    get_epoch_range: obtain start and end epochs of science data set
    get_filters: obtain filters list
    get_ref_index: obtain refernce frame index
    extract_lightcurve: carry out magnitude calculation and calibration and plot lightcurve
    
    Static Methods
    --------------
    inv_median: return the reciprocal of the median value
    distortion_correct: correct for distortion at varying distance caused by telescope
    """
    
    
    def __init__(self, name, input_path, telescope_id):
        """
        Initialise lightcurve data set

        Parameters
        ----------
        name : string
            name of moving object
        input_path : string
            directory for science images
        telescope_id : int
            id number for telescope location in JPL Horizons system
        """
        
        self.name = str(name)
        self.input_path = str(input_path) #path for raw_science_data
        
        #define path for bias and flat images
        path = Path(input_path)
        self.day = str(path.resolve().parents[0])[-2:]
        base_path = str(path.resolve().parents[2])
        self.bias_path = base_path + '/Bias' + self.day 
        self.flat_path = base_path + '/flats'
        
        #define basenames for future naming of images
        science_basenames = []
        science_names = sorted(glob.glob(input_path + '/*.fit'))
        for i in range(len(science_names)):
            science_basenames.append(os.path.basename(science_names[i]))
        self.science_basenames = science_basenames
        
        flat_basenames = []
        flat_names = sorted(glob.glob(self.flat_path + '/*.fit'))
        for i in range(len(flat_names)):
            flat_basenames.append(os.path.basename(flat_names[i]))
        self.flat_basenames = flat_basenames
        
        bias_basenames = []
        bias_names = sorted(glob.glob(self.bias_path + '/*.fit'))
        for i in range(len(bias_names)):
            bias_basenames.append(os.path.basename(bias_names[i]))
        self.bias_basenames = bias_basenames
        
        self.n = 4 #set standard extension to 4 unless overwritten
        self.telescope_id = telescope_id
        self.fwhm = None
        self.pixel_scale = 0.333 #in arcseconds/pixel
        self.ref_frame_index = None
        self.midtime = []
        self.rel_df = None
        self.stellar_objids = None
        self.m = None
        self.m_err = None
        self.cal_med_rel = None
        self.cal_med_rel_err = None
        self.colour = 0.2 #estimate colour of moving object
        self.object_r_mag = []
        self.object_r_mag_err = []
        self.median_rel = None #initiate relative magnitudes 
        self.median_rel_err = None
        self.filters = None
        self.start_time = None
        self.end_time = None
        self.second_filt = None
        self.alpha = None
        self.r = None
        self.delta = None
        self.abs_mag = []
        self.length = None
        self.index = None
        
    def __str__(self):
        
        return f"{self.name}"
        
    def extract_and_overscan(self, biassec, trimsec):
        """
        Run extension extractor and subtract and trim overscan

        Parameters
        ----------
        biassec : string
            Biassec in [x,y] style and fits format
        trimsec : string
            Trimsec in [x,y] style and fits format
        """
        
        #create path for saving images
        if not os.path.exists(self.input_path + '/Reduced/Overscan'):
            os.makedirs(self.input_path + '/Reduced/Overscan')
        

        
        for i in range(len(self.science_basenames)):
            #save only necessary extension and combine headers
            with fits.open(self.input_path + '/' + self.science_basenames[i], mode='readonly') as hdu:
                primary_header   = hdu[0].header
                extension_data   = hdu[self.n].data
                extension_header = hdu[self.n].header
                extension_header += primary_header
                
            fits.writeto(self.input_path + '/Reduced/Overscan/' + self.science_basenames[i], extension_data, extension_header, overwrite=True,
                                    output_verify='fix')
            
            #compute overscan subtraction and trimming
            calib = CCDData.read(self.input_path + '/Reduced/Overscan/' + self.science_basenames[i], unit = 'adu')
            calib = ccdp.subtract_overscan(calib, median=True, overscan_axis=1, fits_section=biassec)
            calib = ccdp.trim_image(calib, fits_section=trimsec)
            calib.write(self.input_path + '/Reduced/Overscan/' + calib.header['OBSTYPE'] + self.science_basenames[i], overwrite=True)
        
        for i in range(len(self.flat_basenames)):
            with fits.open(self.flat_path + '/' + self.flat_basenames[i], mode='readonly') as hdu:
                primary_header   = hdu[0].header
                extension_data   = hdu[self.n].data
                extension_header = hdu[self.n].header
                extension_header += primary_header
                
            fits.writeto(self.input_path + '/Reduced/Overscan/' + self.flat_basenames[i], extension_data, extension_header, overwrite=True,
                                    output_verify='fix')
            calib = CCDData.read(self.input_path + '/Reduced/Overscan/' + self.flat_basenames[i], unit = 'adu')
            calib = ccdp.subtract_overscan(calib, median=True, overscan_axis=1, fits_section=biassec)
            calib = ccdp.trim_image(calib, fits_section=trimsec)
            calib.write(self.input_path + '/Reduced/Overscan/SKY' + self.flat_basenames[i], overwrite=True)
            
        for i in range(len(self.bias_basenames)):
            with fits.open(self.bias_path + '/' + self.bias_basenames[i], mode='readonly') as hdu:
                primary_header   = hdu[0].header
                extension_data   = hdu[self.n].data
                extension_header = hdu[self.n].header
                extension_header += primary_header
                    
            fits.writeto(self.input_path + '/Reduced/Overscan/' + self.bias_basenames[i], extension_data, extension_header, overwrite=True,
                                        output_verify='fix')
            calib = CCDData.read(self.input_path + '/Reduced/Overscan/' + self.bias_basenames[i], unit = 'adu')
            calib = ccdp.subtract_overscan(calib, median=True, overscan_axis=1, fits_section=biassec)
            calib = ccdp.trim_image(calib, fits_section=trimsec)
            calib.write(self.input_path + '/Reduced/Overscan/' + calib.header['OBSTYPE'] + self.bias_basenames[i], overwrite=True)
        
        print("Completed overscan subtraction")

    def master_bias(self, gain, sigma_clip=3.0):
        """
        Combine the bias images using a median averaging and median sigma clip

        Parameters
        ----------
        gain : float
            Gain in electrons/adu
        sigma_clip : float
            optional. Clip value for median sigma clipping

        """
        bias_list = sorted(glob.glob(self.input_path + '/Reduced/Overscan/BIAS*.fit'))
        
        #combine bias images using average and then gain correct
        master_bias = ccdp.combine(bias_list, method = 'average', sigma_clip = True, sigma_clip_low_thresh = sigma_clip, sigma_clip_high_thresh = sigma_clip, sigma_clip_func = np.ma.median, sigma_clip_dev_func = mad_std, mem_limit = 350e6)
        master_bias = ccdp.gain_correct(master_bias, gain, u.electron/u.adu)

        master_bias.meta['combined'] = True

        master_bias.write(self.input_path + '/Reduced/master_bias.fit', overwrite = True)
        
        print("Completed master bias creation")
        
    def inv_median(a):
        """
        Calculate the reciprocal of the median of a given list

        Parameters
        ----------
        a : list

        Returns
        -------
        float
            inverse of median value

        """
        return 1/np.median(a)

    def master_flats(self,  gain, sigma_clip=3.0):
        """
        Bias subtract from flats and then do median combine and median sigma
        clip to create a master flat for each filter

        Parameters
        ----------
        gain : float
            Gain in electrons/adu
        sigma_clip : float
            optional. Clip value for median sigma clipping
        """
        
        flat_list = sorted(glob.glob(self.input_path + '/Reduced/Overscan/SKY*.fit'))
        
        #create paths for saving images
        if not os.path.exists(self.input_path + '/Reduced/Flats'):
            os.makedirs(self.input_path + '/Reduced/Flats')
            
        if not os.path.exists(self.input_path + '/Reduced/Combined_flats'):
            os.makedirs(self.input_path + '/Reduced/Combined_flats')
        
        master_bias = CCDData.read(self.input_path + '/Reduced/master_bias.fit')
        
        for i in range(len(flat_list)): #leave out as was too low
            flat = CCDData.read(flat_list[i])
            flat = ccdp.gain_correct(flat, gain, u.electron/u.adu)
            flat_bias_subtracted = ccdp.subtract_bias(flat, master_bias)
            flat_bias_subtracted.write(self.input_path + '/Reduced/Flats/' + (os.path.basename(flat_list[i]).removeprefix('SKY')), overwrite = True)
        
        
        #group flats by filer type
        flats = ccdp.ImageFileCollection(self.input_path + '/Reduced/Flats')
        flat_filters = set(h['wffband'] for h in flats.headers(obstype = 'SKY'))
        
        #create a master flat for each filter type
        for filt in flat_filters:
            to_combine = flats.files_filtered(obstype = 'SKY', wffband = filt, include_path = True)
            combined_flat = ccdp.combine(to_combine, method = 'median', scale=lightcurve.inv_median, sigma_clip=True, sigma_clip_low_thresh=sigma_clip, sigma_clip_high_thresh=sigma_clip, sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,mem_limit=350e6)
        
            combined_flat.meta['combined'] = True
            flat_file_name = 'combined_flat_filter_{}.fit'.format(filt.replace("''", "p"))
            combined_flat.write(self.input_path + '/Reduced/Combined_flats/' + flat_file_name, overwrite = True)
            
        print("Completed master_flats creation")
        
        
    def science_reduction(self, readnoise, gain):
        """
        Reduce science images using master bias and master flats

        Parameters
        ----------
        readnoise : float
            readnoise of the detector in electrons
        gain : float
            Gain in electrons/adu
        """
        
        science_overscan_list = sorted(glob.glob(self.input_path + '/Reduced/Overscan/TARGET*.fit'))
        master_flats_collection = ccdp.ImageFileCollection(self.input_path + '/Reduced/Combined_flats')
        
        #create path for saving images
        if not os.path.exists(self.input_path + '/Reduced/Science'):
            os.makedirs(self.input_path + '/Reduced/Science')
            
        filters = []
        times = []
        
        for i in range(len(science_overscan_list)):
            science = CCDData.read(science_overscan_list[i])
            filt = science.header['wffband'] #determine which filter flat to use
            filters.append(filt)
            
            times.append(science.header['jd'])
            
            if not os.path.exists(self.input_path + '/Reduced/Science/' + filt):
                os.makedirs(self.input_path + '/Reduced/Science/' + filt)
            
            flat_file_name = master_flats_collection.files_filtered(include_path = True, combined = True, wffband = filt)
            flat_file = CCDData.read(flat_file_name[0], unit = u.electron)
            master_bias = CCDData.read(self.input_path + '/Reduced/master_bias.fit', unit = u.electron)
            science = ccdp.ccd_process(science, master_bias = master_bias, master_flat = flat_file, readnoise = readnoise*u.electron, gain = gain *u.electron/u.adu)
            science.write(self.input_path + '/Reduced/Science/' + filt + '/' + os.path.basename(science_overscan_list[i]).removeprefix('TARGET') , overwrite = True)
        
        self.start_time = str(times[0])
        self.end_time = str(times[-1])
        self.filters = np.unique(np.array(filters))        
            
        print("Completed science reduction")
            
    def plate_solve(self, filt):
        """
        Determine WCS transformations for each image and save to header to plate solve.
        Uses local index files.
        
        Parameters
        ----------
        filt : str
            filter name
        """
        science_images = sorted(glob.glob(self.input_path + '/Reduced/Science/' + filt + '/*[!wcs].fit'))
        
        #create path for saving extra plate solve files
        if not os.path.exists(self.input_path + '/Reduced/Extra'):
            os.makedirs(self.input_path + '/Reduced/Extra')
        
        for i in range(len(science_images)):
            #run solve-field, and change new file to .fit
            science = CCDData.read(science_images[i], unit = u.electron)
            
            #run bash terminal script to solve field
            script_path = str(os.getcwd()) + '/solve_field.sh'
            
            subprocess.call('%s %s %s %s %s %s' % (script_path, (os.path.basename(science_images[i])).removesuffix('.fit'), str(science.header['RA']), str(science.header['DEC']), str(science_images[i]), self.input_path + '/Reduced/Extra'), shell=True)
            wcs_image = glob.glob(self.input_path + '/Reduced/Science/' + filt + '/*wcs.new')
            if len(wcs_image) != 0:
                new = Path(str(wcs_image[0]))
                new.rename(new.with_suffix('.fit'))
            
        print("Completed plate solving")
        
        
    def iterative(self, filt):
        """
        Perform iterative source detection and psf fitting to determine an
        accurate stellar fwhm for each image
        
        Parameters
        ----------
        filt : str
            filter name
        """
        wcs_images = sorted(glob.glob(self.input_path+ '/Reduced/Science/' + filt + '/*wcs.fit'))
        imgs = []
        fwhm = []
        
        #create new name for saving
        for i in range(len(wcs_images)):
            imgs.append(str(wcs_images[i].removesuffix('.fit') + '_iterative.fit'))
        
        for i in range(len(wcs_images)):
            
            #open files
            temp_file = fits.open(wcs_images[i])
            file = temp_file[0]
            img = file.data
            
            pixel_scale = self.pixel_scale
            #output stats
            mean, median, std = sigma_clipped_stats(img, sigma=2.5)  
            
            #Set up psf fitting parmeters and routines
            fwhm_psf=4.0
            sigma_psf=fwhm_psf/2.354
            star_find = DAOStarFinder(fwhm=fwhm_psf, threshold=10.0*std,peakmax=60000.0,sigma_radius=5.0,sharphi=1.0)
        
            daogroup = DAOGroup(9.0)
            med_bkg = MMMBackground()
        
            # define mdel psf as a gaussian
            model_psf = IntegratedGaussianPRF(sigma=sigma_psf)
            model_psf.sigma.fixed = False
        
            # define fitting routine to use
            fitter = LevMarLSQFitter()
        
            # create function to perform psf fitting on frame
            my_photometry = IterativelySubtractedPSFPhotometry(finder=star_find, group_maker=daogroup, bkg_estimator=med_bkg, psf_model=model_psf, fitter=fitter, fitshape=(11,11),niters=2)
            
            #set psf model, run source detection and photometry
            table = my_photometry(image=img-median)
            
            #calculate fwhm values
            table['fwhmarcsec']=pixel_scale*2.354*table['sigma_fit']
            table['fwhmpix']=2.354*table['sigma_fit']
            
            indiv_fwhm = table['fwhmpix'].value
            fwhm.append((np.median(indiv_fwhm))) #get average fwhm for each image
            self.fwhm = fwhm
            
            fwhm_df = pd.DataFrame(data = {'fwhm':fwhm})
            fwhm_df.to_csv(self.input_path + '/' + filt + '_fwhm_list.csv', index = False)
            
            #turn results in to bintable and add to hdulist
            bintable = fits.table_to_hdu(table)
            if len(temp_file) == 1:
                temp_file.append(bintable)
                
            elif len(temp_file)== 2:
                temp_file[1] = bintable
            
            #write file and close
            temp_file.writeto(imgs[i], overwrite=True)
            temp_file.close()
            print('Loop: ' + str(i))
            
        print("Completed iterative photometry and obtained fwhm values")

    def distortion_correct(x,y,flux):
        """
        Correct for telescope distortion depending on distance from optical
        axis

        Parameters
        ----------
        x : float
            x pixel location
        y : float
            y pixel location
        flux : float
            background subtracted flux value for source

        Returns
        -------
        f : float
            corrected flux value

        """
        x0 = 1778 #optical axis x pixel
        y0 = 3029 # optical axis y pixel
        #angular distance from optical axis in radians
        r = np.sqrt((x-x0)**2 + (y-y0)**2)*0.333*np.pi/(180*3600) 
        f = flux/((1+3*220*(r**2))*(1+220*(r**2)))
        
        return f
        
        
    def detect_and_phot(self, filt):
        """
        Run DAOStarFinder to detect all sources in an image. Then complete aperture
        photometry on each source. Convert output results to bintable and save as 
        an extension to the image's fits file. Repeat for all images.
        
        Parameters
        ----------
        filt : str
            filter name
        """
        
        wcs_images = sorted(glob.glob(self.input_path+ '/Reduced/Science/' + filt + '/*wcs.fit'))
        imgs = []
        temp_fwhm = []
        
        #create new name for saving
        for i in range(len(wcs_images)):
            imgs.append(str(wcs_images[i].removesuffix('.fit') + '_aperture.fit'))
        
        for i in range(len(wcs_images)):
            
            #open files
            temp_file = fits.open(wcs_images[i])
            file = temp_file[0]
            img = file.data
            
            mean, median, std = sigma_clipped_stats(img, sigma=2.5)
            
            #setup and run source detection
            starfind = DAOStarFinder(10*std, self.fwhm[i])            
            table = starfind(img)
            
            #define aperture and annuli positions
            positions = np.array([table['xcentroid'].data, table['ycentroid'].data])
            positions = np.swapaxes(positions, 0, 1)
            apertures = CircularAperture(positions, r = self.fwhm[i])
            annuli = CircularAnnulus(positions, r_in=self.fwhm[i]+5, r_out=self.fwhm[i]+10)
            
            #run photometry on apertures            
            sigclip = SigmaClip(sigma = 3.0, maxiters = 10)
            aper_stats = ApertureStats(img, apertures, sigma_clip=None)
            bkg_stats = ApertureStats(img, annuli, sigma_clip=sigclip)
            
            #calculate and save background subtracted flux values
            total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value
            apersum_bkgsub = aper_stats.sum - total_bkg
            aper_stats = aper_stats.to_table()
            aper_stats['bkgsub'] = apersum_bkgsub  #save background subtracted count
            aper_stats['flux_fit'] = apersum_bkgsub #save counts before distortion correction
            aper_stats['sum_err'] = total_bkg
            aper_stats.remove_columns(['sky_centroid'])
            
            #correct for instrument distortion
            for j in range(len(table)):
                aper_stats['flux_fit'][j] = lightcurve.distortion_correct(aper_stats['xcentroid'][j], aper_stats['ycentroid'][j], aper_stats['flux_fit'][j])
            
            #calculate average fwhm for each image
            indiv_fwhm = aper_stats['fwhm'].value
            temp_fwhm.append((np.median(indiv_fwhm)))
            
            #turn results in to bintable and add to hdulist
            bintable = fits.table_to_hdu(aper_stats)
            if len(temp_file) == 1:
                temp_file.append(bintable)
                
            elif len(temp_file)== 2:
                temp_file[1] = bintable
            
            #write file and close
            temp_file.writeto(imgs[i], overwrite=True)
            temp_file.close()
        
        #find frame with best fwhm as a reference frame
        self.ref_frame_index = temp_fwhm.index(min(temp_fwhm))
        ref_index_df = pd.DataFrame(data = {'ref_frame_index': [self.ref_frame_index]})
        ref_index_df.to_csv(self.input_path + '/' + filt + '_ref_frame_index.csv', index = False)
        
            
        print("Completed source detection and photometry")
            

    def ephemerides(self, step = '1m', name = None):
        """
        Query the JPL Horizons database for ephemerids of specified object.
        Then apply a linear fit to the data for RA and Dec at exact times within
        specified time range.

        Parameters
        ----------
        start_time : str
            start time of query in format YYYY-MM-DD HH-mm
        end_time : str
            end time of query in format YYYY-MM-DD HH-mm
        step : str
            optional. time step for ephemerides with unit (d,h,m)
        name : str
            optional. name of moving object, defaults to self.name
        
        Returns
        -------
        m_ra : float
            gradient of best fit line for RA vs time
        c_ra : float
            y-intercept of best fit line for RA vs time
        y_dec : float
            gradient of best fit line for Dec vs time
        c_dec : float
            y-intercept of best fit line for Dec vs time
        """
        
        if name == None:
            name = self.name
        
        start = Time(self.start_time, format = 'jd')
        end = Time(self.end_time, format = 'jd')
        
        #define Horizons object for set times
        object_name = Horizons(id = name, location = self.telescope_id, epochs = {'start':start.iso, 'stop':end.iso, 'step':step })
        eph = object_name.ephemerides()
        
        self.alpha = np.nanmedian(eph['alpha_true']) #get median phase angle
        self.delta = np.nanmedian(eph['delta']) #get distance of object from earth
        self.r = np.nanmedian(eph['r']) #get distance of object from Sun
        
        #create SkyCoord objects
        eph['Coordinates'] = SkyCoord(eph['RA'], eph['DEC'], unit = u.degree)
        
        #create linear fit to ephemerides data to get position at exact times
        x = eph['datetime_jd']
        y_ra = eph['Coordinates'].ra.degree
        m_ra,c_ra = np.polyfit(x, y_ra , 1)
        
        y_dec = eph['Coordinates'].dec.degree
        m_dec,c_dec = np.polyfit(x, y_dec , 1)
        
        print("Completed ephemerides query")
        return m_ra, c_ra, m_dec, c_dec
    
            
    def catalogue_stars(self, m_ra, c_ra, m_dec, c_dec, filt):
        """
        Match all sources to PS1 catalogue. Then pick out only stellar sources.
        Then, determine which stars appear in all images to use for calbration.
        Finally, lookup and store all necessary magnitude data for each star.
        Parameters
        ----------
        m_ra : float
            gradient of best fit line for RA vs time
        c_ra : float
            y-intercept of best fit line for RA vs time
        y_dec : float
            gradient of best fit line for Dec vs time
        c_dec : float
            y-intercept of best fit line for Dec vs time
        filt : str
            filter name
        """
        
        aperture_images = sorted(glob.glob(self.input_path+ '/Reduced/Science/' + filt + '/*wcs_aperture.fit'))
        
        #initiate lists and dataframe
        temp_objids = []
        temp1_objids = []
        midtime = []
        object_r_inst = []
        object_r_inst_err = []
        df = pd.DataFrame({'objid':[]})
        
        self.length = len(aperture_images)
        self.index = []
        
        for i in range(len(aperture_images)):
            
            #open all file data
            image = fits.open(aperture_images[i])
            header = image[0].header
            wcs = WCS(image[0].header)
            phot = Table(image[1].data)
        
            #get catalog
            ps1 = cvc.PanSTARRS1('ps1_cat.db')
            
            #create SkyCoord objects for all sources
            coords = SkyCoord.from_pixel(phot['xcentroid'], phot['ycentroid'], wcs, 0, mode='all')
            
            #calculate exact position of object
            midtime.append(header['jd'] + header['ELAPSED']/(2*24*3600))
            RA = m_ra*midtime[-1] + c_ra
            Dec = m_dec*midtime[-1] + c_dec
            eph_coord = SkyCoord(RA, Dec,  unit=u.degree)
        
            #calculate separations between object and each matched source
            separations = coords.separation(eph_coord)
        
            #download PS1 Catalog
            if len(ps1.search(coords)[0]) < 500:
                ps1.fetch_field(coords)
        
            #create masked array with matched objects
            objids, distances = ps1.xmatch(coords)
            
            if filt == 'r':
                self.second_filt = 'i'
                
            elif (filt == 'g') or (filt == 'i'):
                self.second_filt = 'r'
                
            elif filt == 'z':
                self.second_filt = 'y'
                
            else:
                self.second_filt = 'z'
            
            #define data to lookup from catalogue
            if filt == 'i':
                columns = ['objid', 'imeanpsfmag', 'imeanpsfmagerr', (str(self.second_filt) + 'meanpsfmag'), (str(self.second_filt) + 'meanpsfmagerr'), 'imeankronmag']
            
            elif self.second_filt == 'i':
                columns = ['objid', 'imeanpsfmag', 'imeanpsfmagerr', (str(filt) + 'meanpsfmag'), (str(filt) + 'meanpsfmagerr'), 'imeankronmag']
            
            else:
                columns = ['objid', 'imeanpsfmag', 'imeanpsfmagerr', (str(filt) + 'meanpsfmag'), (str(filt) + 'meanpsfmagerr'), (str(self.second_filt) + 'meanpsfmag'), (str(self.second_filt) + 'meanpsfmagerr'), 'imeankronmag']
            
            #create an array of data looked up
            data = ps1.lookup(objids.data[~objids.mask], ','.join(columns))
            data_array = np.array(data)
            
            #create masked array for each column
            for x, j in zip(columns, range(len(columns))):
                cat_data = np.zeros(len(objids))
                z = 0
                for k in range(len(objids)):
                    if objids.mask[k] == False:
                        cat_data[k] = data_array[z,j]
                        
                        z += 1
                
                globals()[x] = ma.masked_array(cat_data, objids.mask)
            
            #add masked arrays to photometry table
            phot['objid'] = objid
            phot['imeanpsfmag'] = imeanpsfmag
            phot['imeanpsfmagerr'] = imeanpsfmagerr
            phot[(str(filt) + 'meanpsfmag')] = globals()[(str(filt) + 'meanpsfmag')]
            phot[(str(filt) + 'meanpsfmagerr')] = globals()[(str(filt) + 'meanpsfmagerr')]
            phot[(str(self.second_filt) + 'meanpsfmag')] = globals()[(str(self.second_filt) + 'meanpsfmag')]
            phot[(str(self.second_filt) + 'meanpsfmagerr')] = globals()[(str(self.second_filt) + 'meanpsfmagerr')]
            
            #create a dataframe of catalogue data
            sources = pd.DataFrame(data = (list(d) for d in data), columns = columns)
            
            #determine only the stellar objects
            
            if (self.second_filt == 'z') or (self.second_filt  == 'r'):
                star_objids = sources[(sources['imeanpsfmag']!=-999) & (sources['imeankronmag']!=-999) & ((sources['imeanpsfmag']-sources['imeankronmag'])<0.05) & (sources['imeanpsfmag']>=14) & (sources['imeanpsfmag']<=21) & ((sources[(str(self.second_filt) + 'meanpsfmag')] - sources[(str(filt) + 'meanpsfmag')]) <=1.0)]['objid']
            else:
                star_objids = sources[(sources['imeanpsfmag']!=-999) & (sources['imeankronmag']!=-999) & ((sources['imeanpsfmag']-sources['imeankronmag'])<0.05) & (sources['imeanpsfmag']>=14) & (sources['imeanpsfmag']<=21) & ((sources[(str(filt) + 'meanpsfmag')]-sources[(str(self.second_filt) + 'meanpsfmag')]) <=1.0)]['objid']
            
            #calculate instrumental magnitude and error for each star
            r_inst = -2.5*np.log10(phot['flux_fit'])
            r_inst_err = np.sqrt((np.array(phot['sum']) + np.array(phot['sum_err'])))/np.array(phot['bkgsub']) * 1.0857
            
            #define the brightest non saturated star and calculate relative
            #magnitudes for each star
            non_saturated = r_inst[phot['max']<184800]
            brightest_unsat = np.nanmin(non_saturated)
            brightest_unsat_err = r_inst_err[r_inst == np.nanmin(non_saturated)]
            bright_diff = brightest_unsat - r_inst
            bright_diff_err = np.sqrt(brightest_unsat_err**2 + r_inst_err**2)
            
            #save magnitudes to photometry table
            phot['r_inst'] = r_inst
            phot['r_inst_err'] = r_inst_err
            
            #find which source is the object
            object_index = list(separations).index(min(separations))
            
            #save object magnitude and error
            object_r_inst.append(phot['r_inst'][object_index])
            object_r_inst_err.append(phot['r_inst_err'][object_index])
            
            if phot['flux_fit'][object_index] <= 0:
                self.length -= 1
                midtime.pop(-1)
                if i <= self.ref_frame_index:
                    self.ref_frame_index -= 1
                    
                continue
            
            self.index.append(i)
            
            temp_objids += list(star_objids)
            
            #calculate differential magnitude of each asteroid and star
            diff_mag = object_r_inst[i]-r_inst
            diff_mag_err = np.sqrt(object_r_inst_err[i]**2 + r_inst_err**2)
            
            #create a data frame with all masked array data for diff photometry
            temp_df = pd.DataFrame(data = {'objid': objid, str('r_inst' + str(i)): r_inst, str('r_inst_err' + str(i)): r_inst_err, str('bright_diff' + str(i)): bright_diff, str('bright_diff_err' + str(i)) : bright_diff_err, str('diff_mag' + str(i)): diff_mag, str('diff_mag_err' + str(i)): diff_mag_err})
            temp_df.dropna(subset = 'objid', inplace = True)
            temp_df.drop_duplicates(subset=['objid'], keep='first', inplace=True, ignore_index=True)

            df = pd.merge(df, temp_df, how = 'outer', on = ['objid'])
            df.dropna(how = 'any', inplace = True) #remove rows with missing values
            
            #save table to hdu
            bintable = fits.table_to_hdu(phot)
            if len(image) == 1:
                image.append(bintable)
                
            elif len(image)== 2:
                image[1] = bintable
            
            #write file and close
            image.writeto(aperture_images[i], overwrite=True)
            image.close()
            print('Loop:' + str(i))

        #create an array of only star objids which appear in all images
        for i in range(len(temp_objids)):
            if op.countOf(temp_objids, temp_objids[i]) == self.length:
                temp1_objids.append(temp_objids[i])
        stellar_objids = np.unique(temp1_objids)
        
        #reset indices for later use
        df.reset_index(level = 0, inplace=True)

        df['mean_bright_diff'] = df['mean_bright_diff_err']= 0
        for i in range(len(df)):
            
            #get the bright star relative magnitudes and errors from dataframe
            bright_diffs = df.iloc[i, list(range(4, 6*self.length + 1, 6))]
            bright_diff_errs = df.iloc[i, list(range(5, 6*self.length + 1, 6))]
            
            df['mean_bright_diff'][i] = np.nanmean(bright_diffs)
            
            #calculate sum of squares of errors
            sum_err = 0
            for j in np.array(bright_diff_errs):
                sum_err += j**2
            
            #propagate the mean error
            df['mean_bright_diff_err'][i] = np.sqrt(sum_err)/self.length
            
            print('Second Loop: ' + str(i))
            
        #create dataframe with only stars
        rel_df = pd.DataFrame({'objid':stellar_objids})    
        rel_df = pd.merge(rel_df, df, on = 'objid')
        
        #scale relative magnitudes of each object by non saturated rel magnitudes                                   
        for i in self.index:
            rel_df[str('rel_mag' + str(i))] = np.array(rel_df[str('diff_mag' + str(i))]) - np.array(rel_df['mean_bright_diff'])
            rel_df[str('rel_mag_err' + str(i))] = np.sqrt(np.array(rel_df[str('diff_mag_err' + str(i))])**2 + np.array(rel_df['mean_bright_diff_err'])**2)            

        self.rel_df = rel_df
        self.midtime.append(midtime)
        self.stellar_objids = stellar_objids
        
        print("Completed star matching")
    
    def calibration(self, m_ra, c_ra, m_dec, c_dec, filt):
        """
        Calibrate the reference frame (image with the lowest median fwhm) using
        orthogonal distance regression modelling and calculate a calibrated
        object magnitude in that frame. Also, calculate the object relative
        magnitude in the reference frame for later comparison.

        Parameters
        ----------
        m_ra : float
            gradient of best fit line for RA vs time
        c_ra : float
            y-intercept of best fit line for RA vs time
        y_dec : float
            gradient of best fit line for Dec vs time
        c_dec : float
            y-intercept of best fit line for Dec vs time
        filt : str
            filter name

        """
        
        aperture_images = sorted(glob.glob(self.input_path+ '/Reduced/Science/' + filt + '/*wcs_aperture.fit'))
        
        #open all file data
        image = fits.open(aperture_images[self.ref_frame_index])
        header = image[0].header
        wcs = WCS(image[0].header)
        phot = Table(image[1].data)
        
        #create SkyCoord objects for all sources
        coords = SkyCoord.from_pixel(phot['xcentroid'], phot['ycentroid'], wcs, 0, mode='all')
        
        #calculate exact position of asteroid
        midtime = (header['jd'] + header['ELAPSED']/(2*24*3600))
        RA = m_ra*midtime + c_ra
        Dec = m_dec*midtime + c_dec
        eph_coord = SkyCoord(RA, Dec,  unit=u.degree)
            
        #calculate coordinates and separations
        phot['Coordinates'] = coords
        phot['separations'] = coords.separation(eph_coord)
        
        #define columns for data lookup
        columns = ['objid', 'imeanpsfmag', 'imeanpsfmagerr', (str(filt) + 'meanpsfmag'), (str(filt) + 'meanpsfmagerr'), (str(self.second_filt) + 'meanpsfmag'), (str(self.second_filt) + 'meanpsfmagerr'),'flux_fit', 'sum_err', 'sum', 'bkgsub']
        
        #create masked array of objids
        objid_original = ma.masked_invalid(np.array(phot['objid']))
        
        #obtain all data for comparison stars
        for x, j in zip(columns, range(len(columns))):
            globals()[x] = ma.masked_invalid(np.array(phot[x]))
            mask_stars = np.isin(objid_original, self.stellar_objids)
            globals()[x] = ma.masked_array(globals()[x], ~mask_stars)
            globals()[x] = (globals()[x][globals()[x].mask == False]).data
            
        #calculate instrumental magnitudes and errors
        r_inst = -2.5*np.log10(flux_fit)
        r_inst_err = np.sqrt((np.array(sum) + sum_err))/bkgsub * 1.0857
        
        #calculate colour and differential magnitudes
        if (self.second_filt == 'z') or (self.second_filt  == 'r'):
            x = (globals()[(str(self.second_filt) + 'meanpsfmag')]) - (globals()[(str(filt) + 'meanpsfmag')]) 
        
        else:
            x = (globals()[(str(filt) + 'meanpsfmag')]) - (globals()[(str(self.second_filt) + 'meanpsfmag')])
        
        x_err = np.sqrt((globals()[(str(filt) + 'meanpsfmagerr')])**2 + (globals()[(str(self.second_filt) + 'meanpsfmagerr')])**2)
        y = (globals()[(str(filt) + 'meanpsfmag')]) - r_inst
        y_err = np.sqrt((globals()[(str(filt) + 'meanpsfmagerr')])**2 + r_inst_err**2)
        
        #apply orthogonal distance regression to get a colour term and initial zp
        data = odr.RealData(x, y, sx = x_err, sy = y_err)
        odr1 = odr.ODR(data, odr.unilinear)
        out = odr1.run()
        C = out.beta[0]
        zp_initial = out.beta[1]
        C_err = out.sd_beta[0]
        
        #plot data and initial calibration fit
        plt.scatter(x,y)
        x1 = np.linspace(min(x),max(x))
        plt.plot(x1, C*x1 + zp_initial, 'r')
        
        #move line to all stars and get zp's
        intercept_list = []
        for i in range(len(x)):
            intercept_list.append(y[i]-C*x[i])
        
        #calculate final zp and plot
        zp = np.nanmedian(intercept_list)
        zp_err = np.nanstd(intercept_list)
        plt.plot(x1, C*x1 + zp, 'g')
        
        #assume colour, then calculate magnitude and relative magnitudes
        colour = self.colour
        object_mask = phot['separations'] == min(phot['separations'])
        
        m_inst = float(phot['r_inst'][object_mask])
        m = C*colour + zp + m_inst
        m_inst_err = float(phot['r_inst_err'][object_mask])
        m_err = np.sqrt((colour**2)*(C_err**2) + (m_inst_err**2) + (zp_err**2))
        
        median_rel = []
        median_rel_err = []
        
        #calculate median and error of relative magnitudes
        for i,z in zip(self.index, range(self.length)):
            median_rel.append(np.nanmedian(np.array(self.rel_df.loc[:, (str('rel_mag'+str(i)))])))
            
            if (len(self.rel_df) % 2) == 0:
                j = int(len(self.rel_df)/2)
                new_data = sorted(list(self.rel_df.loc[:, (str('rel_mag' + str(i)))]))
                l = list(self.rel_df.loc[:, (str('rel_mag' + str(i)))]).index(new_data[j-1])
                up = list(self.rel_df.loc[:, (str('rel_mag' + str(i)))]).index(new_data[j])
                med_err = 0.5*np.sqrt(self.rel_df.loc[l, (str('rel_mag_err' + str(i)))]**2 + self.rel_df.loc[up, (str('rel_mag_err' + str(i)))]**2)
        
            else:
                median_index = list(self.rel_df.loc[:, (str('rel_mag' + str(i)))]).index(median_rel[z])
                med_err = self.rel_df.loc[:, (str('rel_mag_err' + str(i)))][median_index]
                
            med_std = np.nanstd(np.array(self.rel_df.loc[:, (str('rel_mag'+str(i)))]))
            
            median_rel_err.append(np.sqrt(med_err**2 + med_std**2))
        
        self.median_rel, self.median_rel_err = np.array(median_rel), np.array(median_rel_err)
        
        #get relative magnitude for reference frame
        cal_med_rel = median_rel[self.ref_frame_index]
        cal_med_rel_err = median_rel_err[self.ref_frame_index]  
        
        self.m = m
        self.m_err = m_err
        self.cal_med_rel = cal_med_rel
        self.cal_med_rel_err = cal_med_rel_err
        
        print("Completed calibration")
        
        
    def magnitudes(self):
        """
        Calculate relative object magnitude for each image and compare to
        reference frame relative magnitude. Then add the difference to the
        reference frame object magnitude.
        """
        
        self.object_r_mag.append(self.median_rel - self.cal_med_rel + self.m)
    
        self.object_r_mag_err.append(np.sqrt(self.median_rel_err**2 + self.cal_med_rel_err**2 + self.m_err**2))
        
        self.abs_mag.append(self.median_rel - self.cal_med_rel + self.m - 5*np.log10(self.delta * self.r))
        
        print("Completed magnitude calculation")
        
    def save_results(self, name = None):
        """
        Save a results dataframe.

        Parameters
        ----------
        name : str
            optional. Name of object to search for. Defaults to class object name.

        """
        
        if name == None:
            name = self.name
        
        results = pd.DataFrame(data = {'alpha': [self.alpha]})
        for filt in self.filters:
            filt_index = list(self.filters).index(filt)
            filt_results = pd.DataFrame(data = {(str(filt) + '_midtime'): self.midtime[filt_index], (str(filt) + '_object_mag'): self.object_r_mag[filt_index], (str(filt) + '_object_mag_err'): self.object_r_mag_err[filt_index], (str(filt) + '_object_absolute_mag'): self.abs_mag[filt_index]})
            results = pd.concat([results, filt_results], axis = 1)
            
        results.to_csv(self.input_path + '/' + name + '_results.csv', index = False)
    
    def get_fwhm(self, filt):
        """
        Obtain fwhm. Only needed if started after iterative method.

        Parameters
        ----------
        filt : str
            filter name
        """
        fwhm = pd.read_csv(self.input_path + '/' + filt + '_fwhm_list.csv')
        self.fwhm = list(fwhm['fwhm'])
    
    def search_field(self, radius, img_number = None):
        """
        Use Skybot imcce package to query for all moving objects in an image

        Parameters
        ----------
        radius : float
            search radius in arcminutes
        img_number : int, optional
            image number for field search. The default is None.

        Returns
        -------
        results : astropy Table
            table of results sorted by V magnitude

        """
        science_overscan_list = sorted(glob.glob(self.input_path + '/Reduced/Overscan/TARGET*.fit'))
        
        #use middle image if not already defined
        if img_number == None:
            img_number = int(len(science_overscan_list)/2)
            
        image = fits.open(science_overscan_list[img_number])
        header = image[0].header
        
        #get telescope pointing
        field = SkyCoord(header['RA'], header['DEC'], unit= (u.hourangle, u.deg))
        
        #get epoch of image
        epoch = Time(header['jd'] + header['ELAPSED']/(2*24*3600), format='jd')
        
        #search sky for moving objects and convert to astropy table
        temp_results = Skybot.cone_search(field, radius*u.arcmin, epoch)
        results = Table(temp_results)
        results.sort('V')
        results['epoch'] = epoch.jd
        
        return results
    
    def plot_lightcurve(self, filt, lower_ylim = -50, upper_ylim = 75, errors = True, clip = False, name = None):
        """
        Create a lightcurve from given magnitudes, times and errors

        Parameters
        ----------
        filt : str
            filter name
        lower_ylim : float, optional
            lower limit for magnitude value, below which sources will not be displayed. The default is -50.
        upper_ylim : float, optional
            upper limit for magnitude value, above which sources will not be displayed. The default is 75.
        errors : Bool, optional
            Determines whether errors are displayed or not. The default is True.
        clip : Bool, optional
            Determines whether to manually set y limits based on uupper and lower ylim. The default is False.
        name : str
            optional. Name of object to search for. Defaults to class object name.

        Returns
        -------
        fig : matplotlib.pyplot plot
            Calibrated lightcurve

        """
        
        if name == None:
            name = self.name
        
        filt_index = list(self.filters).index(filt)
        
        fig = plt.figure()
        
        #get data for specific filter
        mag_r = np.array(self.object_r_mag[filt_index])
        midtime = np.array(self.midtime[filt_index])
        mag_r_err = np.array(self.object_r_mag_err[filt_index])
        
        #remove all outliers
        mask = (mag_r <upper_ylim) & (mag_r>lower_ylim)
        mag = mag_r[mask]
        time = midtime[mask]
        m_err1 = mag_r_err[mask]
        
        plt.scatter(time, mag)
        
        if errors == True:
            plt.errorbar(time, mag, yerr=m_err1, fmt='o')
            
        ax = plt.gca()
        ax.set_xlabel('JD')
        ax.set_ylabel('r_mag')
        
        if clip == True:
            ax.set_ylim(lower_ylim,upper_ylim)
        
        ax.invert_yaxis() #invert y axis to account for negative magnitude system
        
        fig.savefig(self.input_path + '/' + self.day + '_' + name + '_' + filt + '_lightcurve.png')
        return fig
    
    def full_data_reduction(self, biassec, trimsec, gain, readnoise, step = '1m', sigma_clip=3.0, colour = None):
        """
        Complete all necessary reduction and photometry steps for a lightcurve

        Parameters
        ----------
        biassec : string
            Biassec in [x,y] style and fits format
        trimsec : string
            Trimsec in [x,y] style and fits format
        gain : float
            Gain in electrons/adu
        readnoise : float
            readnoise of the detector in electrons
        step : str
            optional. Time step for ephemerides with unit (d,h,m)
        sigma_clip : float
            optional. Clip value for median sigma clipping
        colour: list
            optional. List of colour for each filter. Defaults to 0.2 for all filters
        """
        self.extract_and_overscan(biassec, trimsec)
        self.master_bias(gain, sigma_clip)
        self.master_flats(gain, sigma_clip)
        self.science_reduction(readnoise, gain)
        
        for filt in self.filters:
            if colour != None:
                self.colour = colour[list(self.filters).index(filt)]
            self.plate_solve(filt)
            self.iterative(filt)
            self.detect_and_phot(filt)
            m_ra, c_ra, m_dec, c_dec = self.ephemerides(step = step)
            self.catalogue_stars(m_ra, c_ra, m_dec, c_dec, filt)
            self.calibration(m_ra, c_ra, m_dec, c_dec, filt)
            self.magnitudes()
            fig = self.plot_lightcurve(filt)
        
        self.save_results()
        
        print("Completed data reduction")
        
            
    def get_epoch_range(self):
        """
        Obtain start and end time of observing. Only needed if starting after 
        science reduction method.
        """
        #open first and last file in science data set
        start = fits.open(self.input_path + '/' + self.science_basenames[0])
        end = fits.open(self.input_path + '/' + self.science_basenames[-1])
        
        #get epochs in JD format
        start_header = start[0].header
        self.start_time = start_header['jd']
        
        end_header = end[0].header
        self.end_time = end_header['jd']
        
        start.close()
        end.close()
        
    
    def get_filters(self):
        """
        Obtain filter list. Only needed if started after science reduction
        method.
        """
        
        science_overscan_list = sorted(glob.glob(self.input_path + '/Reduced/Overscan/TARGET*.fit'))
            
        filters = []
        
        for i in range(len(science_overscan_list)):
            science = CCDData.read(science_overscan_list[i])
            filt = science.header['wffband'] #determine which filter flat to use
            
            filters.append(filt)

        self.filters = np.unique(np.array(filters)) 
        
        
    def get_ref_index(self, filt):
        """
        Obtain reference frame index. Only needed if started after detect and
        phot method.

        Parameters
        ----------
        filt : str
            filter name
        """
        ref_df = pd.read_csv(self.input_path + '/' + filt + '_ref_frame_index.csv')
        self.ref_frame_index = list(ref_df['ref_frame_index'])[0]
    
    def extract_lightcurve(self, step = '1m', name = None, colour = None):
        """
        Complete lightcurve extraction steps if reduction and aperture
        photometry has already been done

        Parameters
        ----------
        step : str
            optional. Time step for ephemerides with unit (d,h,m)
        name : str
            optional. Name of object to search for. Defaults to class object name.
        colour: list
            optional. List of colour for each filter. Defaults to 0.2 for all filters
        """
        #set name to class object name if not specified
        if name == None:
            name = self.name
        
        self.object_r_mag = []
        self.object_r_mag_err = []
        self.midtime = []
        
        self.get_filters()
            
        for filt in self.filters:
            if colour != None:
                self.colour = colour[list(self.filters).index(filt)]
            self.get_epoch_range()
            self.get_ref_index(filt)
            m_ra, c_ra, m_dec, c_dec = self.ephemerides(step = step, name = name)
            self.catalogue_stars(m_ra, c_ra, m_dec, c_dec, filt)
            self.calibration(m_ra, c_ra, m_dec, c_dec, filt)
            self.magnitudes()
            fig = self.plot_lightcurve(filt, name = name)
            
        self.save_results(name)
        
    def extract_single_lightcurve(self, filt, step = '1m', name = None, colour = None):
        """
        Complete lightcurve extraction steps if reduction and aperture
        photometry has already been done

        Parameters
        ----------
        step : str
            optional. Time step for ephemerides with unit (d,h,m)
        name : str
            optional. Name of object to search for. Defaults to class object name.
        colour: list
            optional. List of colour for each filter. Defaults to 0.2 for all filters
        """
        #set name to class object name if not specified
        if name == None:
            name = self.name
        
        self.object_r_mag = []
        self.object_r_mag_err = []
        self.midtime = []
        
            

        if colour != None:
            self.colour = colour[list(self.filters).index(filt)]
        self.get_epoch_range()
        self.get_ref_index(filt)
        m_ra, c_ra, m_dec, c_dec = self.ephemerides(step = step, name = name)
        self.catalogue_stars(m_ra, c_ra, m_dec, c_dec, filt)
        self.calibration(m_ra, c_ra, m_dec, c_dec, filt)
        self.magnitudes()
        fig = self.plot_lightcurve(filt, name = name)
            
        self.save_results(name)
        
    def colour_estimate(self, name = None):
        self.get_filters()
        
        if name == None:
            name = self.name
        
        results = pd.read_csv(self.input_path + '/' + name + '_results.csv')
        
        colour_df = pd.DataFrame(data = {'filter': [], 'midtime': [], 'mag': [], 'mag_err': []})
        for filt in self.filters:
            temp_df = pd.DataFrame(data = {'filter': str(filt), 'midtime': np.array(results[(str(filt) + '_midtime')]), 'mag': np.array(results[(str(filt) + '_object_mag')]), 'mag_err': np.array(results[(str(filt) + '_object_mag_err')])})
            colour_df = pd.concat([colour_df, temp_df])
        
        colour_df.dropna(how = 'any', inplace = True)
        colour_df.sort_values(by= ['midtime'], inplace = True)
        
        colour_df.reset_index(level = 0, inplace=True)
        
        colour_indices = []
        
        for i in range(len(colour_df)-1):
            filter1 = str(colour_df.loc[i, 'filter'])
            filter2 = str(colour_df.loc[i+1, 'filter'])
            if (filter2 + '-' + filter1) in colour_df.columns:
                colour_df.loc[i, (filter2 + '-' + filter1)] = colour_df.loc[i+1, 'mag'] - colour_df.loc[i, 'mag']
                colour_df.loc[i, (filter2 + '-' + filter1 + '_err')] = colour_df.loc[i+1, 'mag_err'] - colour_df.loc[i, 'mag_err']
            
            elif not (str(colour_df.loc[i, 'filter']) + '-' + str(colour_df.loc[i+1, 'filter'])) in colour_df.columns:
                colour_df[(filter1 + '-' + filter2)] = np.nan
                colour_df[(filter1 + '-' + filter2 + '_err')] = np.nan
                
                colour_indices.append((filter1 + '-' + filter2))
                
                colour_df.loc[i, (filter1 + '-' + filter2)] = colour_df.loc[i, 'mag'] - colour_df.loc[i+1, 'mag']
                colour_df.loc[i, (filter1 + '-' + filter2 + '_err')] = colour_df.loc[i, 'mag_err'] - colour_df.loc[i+1, 'mag_err']
            
            else:
                colour_df.loc[i, (filter1 + '-' + filter2)] = colour_df.loc[i, 'mag'] - colour_df.loc[i+1, 'mag']
                colour_df.loc[i, (filter1 + '-' + filter2 + '_err')] = colour_df.loc[i, 'mag_err'] - colour_df.loc[i+1, 'mag_err']
        
        colour_df['colour_index'] = np.nan
        colour_df['mean_colour'] = np.nan
        colour_df['mean_colour_err'] = np.nan
        
        for i in range(len(colour_indices)):
            colour_df.loc[i, 'colour_index'] = colour_indices[i]
            colour_df.loc[i, 'mean_colour'] = np.nanmean(np.array(colour_df[colour_indices[i]]))
            
            errors = np.array(colour_df[(colour_indices[i] + '_err')])
            errors = errors[~np.isnan(errors)]
                              
            sum_err = 0
            for j in errors:
                sum_err += j**2
            
            #propagate the mean error
            colour_df.loc[i, 'mean_colour_err'] = np.sqrt(sum_err)/len(errors)
    
        colour_df.to_csv(self.input_path + '/' + name + '_colour.csv', index = False)
