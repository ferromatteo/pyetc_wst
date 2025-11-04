import logging
import os, sys
import numpy as np

from mpdaf.obj import Spectrum, WaveCoord
from mpdaf.log import setup_logging

from .etc import ETC, get_data

# used by get_data
from astropy.table import Table
import astropy.units as u

CURDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SKYDIR = CURDIR + '/sky'
WSTDIR = CURDIR + '/wst'

class WST(ETC):
    
    def __init__(self, log=logging.INFO, skip_dataload=False):
        self.refdir = CURDIR
        setup_logging(__name__, level=log, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        
        # ------ Telescope ---------
        self.name = 'WST'
        
        self.tel = dict(effective_area_MOS=93.57, # mean of median and weighted mean of the ICD document
                        effective_area_IFS=92.03, # minimum of the ICD document
                        diameter=12.0, # primary diameter
                        desc='Cass design',
                        version='03/11/2025',
                        )        
        # ------- IFS -----------
        self.ifs = {} 
        self.ifs['channels'] = ['blue', 'red']
        # IFS blue channel
        chan = 'blue'
        self.ifs[chan] = dict(desc = 'Inspired from BlueMUSE throughput, modified with inputs from the IFS team',
                              version = '15/10/2025',
                              type = 'IFS',
                              iq_fwhm_tel = np.sqrt(2)/2 * 0.10, # fwhm PSF of telescope **
                              iq_fwhm_ins = 0.30, # fwhm PSF of instrument
                              iq_beta = 2.50, # beta PSF of telescope + instrument
                              spaxel_size = 0.25, # spaxel size in arcsec
                              dlbda = 0.50, # Angstroem/pixel
                              lbda1 = 3700, # starting wavelength in Angstroem
                              lbda2 = 6400, # end wavelength in Angstroem
                              lsfpix = 3.0, # LSF in spectel
                              ron = 3.0, # readout noise (e-)
                              dcurrent = 3.0, # dark current (e-/pixel/h)                                
                              )
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', SKYDIR, WSTDIR)

        # IFS red channel
        chan = 'red'
        self.ifs[chan] = dict(desc='Inspired from MUSE throughput, modified with inputs from the IFS team', 
                               version = '15/10/2025',
                               type='IFS',
                               iq_fwhm_tel = np.sqrt(2)/2 * 0.10, # fwhm PSF of telescope **
                               iq_fwhm_ins = 0.30, # fwhm PSF of instrument
                               iq_beta = 2.50, # beta PSF of telescope + instrument
                               spaxel_size = 0.25, # spaxel size in arcsec
                               dlbda = 0.67, # Angstroem/pixel
                               lbda1 = 6200, # starting wavelength in Angstroem
                               lbda2 = 9800, # end wavelength in Angstroem
                               lsfpix = 3.0, # LSF in spectel
                               ron = 3.0, # readout noise (e-)
                               dcurrent = 3.0, # dark current (e-/pixel/h)
                               )
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', SKYDIR, WSTDIR)
              
        # # --------- MOSLR-VIS 3 channels 6k CCD -------------
        
        # # # update with these values
        # # # https://stfc365.sharepoint.com/:w:/r/sites/Wide-FieldSpectroscopicTelescope/_layouts/15/Doc.aspx?sourcedoc=%7B88FA295F-6BE1-4C0E-8885-1856DE7B8383%7D&file=MOS-LR%20ETCinputs_v02.docx&action=default&mobileredirect=true
        self.moslr = {} 
        self.moslr['channels'] = ['blue','green', 'red']       
        # MOS-LR blue channel 
        chan = self.moslr['channels'][0]
        self.moslr[chan] = dict(desc='Inspired from 4MOST LR throughput, modified with inputs from the MOSLR team',
                                version = '15/10/2025',
                                type = 'MOS',
                                iq_fwhm_tel = 0.30, # fwhm PSF of telescope **
                                iq_fwhm_ins = 0.30, # fwhm PSF of instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.208, # spaxel size in arcsec
                                aperture = 1.03, # fiber diameter in arcsec
                                dlbda = 0.256, # Angstroem/pixel
                                lbda1 = 3700, # starting wavelength in Angstroem
                                lbda2 = 5178, # end wavelength in Angstroem
                                lsfpix = 4.83, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moslr, chan, 'moslr', SKYDIR, WSTDIR)
            
        # MOS-LR green channel      
        chan = self.moslr['channels'][1] 
        self.moslr[chan] = dict(desc='Inspired from 4MOST LR throughput, modified with inputs from the MOSLR team',
                                version = '15/10/2025',
                                type = 'MOS',
                                iq_fwhm_tel = 0.30, # fwhm PSF of telescope **
                                iq_fwhm_ins = 0.30, # fwhm PSF of instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.208, # spaxel size in arcsec
                                aperture = 1.03, # fiber diameter in arcsec
                                dlbda = 0.352, # Angstroem/pixel
                                lbda1 = 5025, # starting wavelength in Angstroem
                                lbda2 = 7140, # end wavelength in Angstroem
                                lsfpix = 4.83, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moslr, chan, 'moslr', SKYDIR, WSTDIR)


        # MOS-LR red channel      
        chan = self.moslr['channels'][2] 
        self.moslr[chan] = dict(desc='Inspired from 4MOST LR throughput, modified with inputs from the MOSLR team',
                                version = '15/10/2025',
                                type = 'MOS',
                                iq_fwhm_tel = 0.30, # fwhm PSF of telescope **
                                iq_fwhm_ins = 0.30, # fwhm PSF of instrument
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.208, # spaxel size in arcsec
                                aperture = 1.03, # fiber diameter in arcsec
                                dlbda = 0.486, # Angstroem/pixel
                                lbda1 = 6929, # starting wavelength in Angstroem
                                lbda2 = 9700, # end wavelength in Angstroem
                                lsfpix = 4.83, # LSF in spectel
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moslr, chan, 'moslr', SKYDIR, WSTDIR)

            
        # --------- MOS-HR 4 channels -------------
        self.moshr = {} 
        self.moshr['channels'] = ['U','B','V','I']       
        # MOS-HR U channel 
        chan = self.moshr['channels'][0]
        self.moshr[chan] = dict(desc='WST HR spectrometer possible baseline description, renamed B channel',  
                                version = '02/11/2025',
                                type = 'MOS',
                                iq_fwhm_tel = 0.30, # fwhm PSF of telescope **
                                iq_fwhm_ins = 0.30, # fwhm PSF of instrument **
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.137, # spaxel size in arcsec, from catadioptric design, 0.116 for dioptric
                                aperture = 1.00, # fiber diameter in arcsec
                                dlbda = 0.036, # Angstroem/pixel, from catadioptric design, 0.030 for dioptric
                                lbda1 = 3700, # starting wavelength in Angstroem
                                lbda2 = 4090, # end wavelength in Angstroem
                                lsfpix = 2.7, # LSF in spectel, from catadioptric design, 3.6 for dioptric
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', SKYDIR, WSTDIR)
            
        # MOS-HR B channel 
        chan = self.moshr['channels'][1]
        self.moshr[chan] = dict(desc='WST HR spectrometer possible baseline description, renamed Green channel',  
                                version = '02/11/2025',
                                type = 'MOS',
                                iq_fwhm_tel = 0.30, # fwhm PSF of telescope **
                                iq_fwhm_ins = 0.30, # fwhm PSF of instrument **
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.137, # spaxel size in arcsec, from catadioptric design, 0.116 for dioptric
                                aperture = 1.00, # fiber diameter in arcsec
                                dlbda = 0.042, # Angstroem/pixel, from catadioptric design, 0.032 for dioptric
                                lbda1 = 4270, # starting wavelength in Angstroem
                                lbda2 = 4720, # end wavelength in Angstroem
                                lsfpix = 2.7, # LSF in spectel, from catadioptric design, 3.6 for dioptric
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', SKYDIR, WSTDIR)

        # MOS-HR V channel
        chan = self.moshr['channels'][2]
        self.moshr[chan] = dict(desc='WST HR spectrometer possible baseline description, renamed Yellow channel',  
                                version = '02/11/2025',
                                type = 'MOS',
                                iq_fwhm_tel = 0.30, # fwhm PSF of telescope **
                                iq_fwhm_ins = 0.30, # fwhm PSF of instrument **
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.137, # spaxel size in arcsec, from catadioptric design, 0.116 for dioptric
                                aperture = 1.00, # fiber diameter in arcsec
                                dlbda = 0.054, # Angstroem/pixel, from catadioptric design, 0.048 for dioptric
                                lbda1 = 5890, # starting wavelength in Angstroem
                                lbda2 = 6510, # end wavelength in Angstroem
                                lsfpix = 2.7, # LSF in spectel, from catadioptric design, 3.6 for dioptric
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', SKYDIR, WSTDIR)

        # MOS-HR I channel
        chan = self.moshr['channels'][3]
        self.moshr[chan] = dict(desc='WST HR spectrometer possible baseline description, renamed Red channel',  
                                version = '02/11/2025',
                                type = 'MOS',
                                iq_fwhm_tel = 0.30, # fwhm PSF of telescope **
                                iq_fwhm_ins = 0.30, # fwhm PSF of instrument **
                                iq_beta = 2.50, # beta PSF of telescope + instrument
                                spaxel_size = 0.2067, # spaxel size in arcsec
                                aperture = 1.00, # fiber diameter in arcsec
                                dlbda = 0.063, # Angstroem/pixel, from catadioptric design, 0.055 for dioptric
                                lbda1 = 6650, # starting wavelength in Angstroem
                                lbda2 = 7350, # end wavelength in Angstroem
                                lsfpix = 2.7, # LSF in spectel, from catadioptric design, 3.6 for dioptric
                                ron = 3.0, # readout noise (e-)
                                dcurrent = 3.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', SKYDIR, WSTDIR)
        
    def info(self, ins=None):
        if ins is None:
            self._info(['ifs', 'moslr', 'moshr'])
        else:
            self._info([ins])

# # # # # # # MORE # # # # # #
# MOS-HR missing the iq instrument (used 0.3" constant), MOS-LR IQ not clear (used 0.3" constant)
# telescope IQ missing everywhere (used 0.3" constant)
# Moffat Beta missing everywhere (used 2.5 constant)
# Diameter missing everywhere (used 12m constant)
# To update:
# new names of UBVI channels for MOS-HR: U->B, B->Green, V->Yellow, I->Red
# # # # # # # # # # # # # # #
                
           
            

               
        
        

