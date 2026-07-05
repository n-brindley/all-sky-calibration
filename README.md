# all-sky-calibration
Various Python codes to convert azimuth and elevation to image x,y coordinates (and back) for all-sky cameras. Prior star calibration is required. Example star calibration parameters are included in calibration_utils.py and STAR_CAL.py for the UNIS narrow FOV sCMOS camera at KHO, Svalbard, for 2016/01/11 at 19:01:00 UT.

calibration_utils.py only really works if the camera optical axis is close to the geographic zenith (it assumes they are aligned).

STAR_CAL.py is more robust, and includes the whole pipeline for getting the star (x,y) coordinates in the image (recommended to use this one).

Thanks to UNIS and Noora Partamies for providing the example all-sky image.

The SAO star catalogue is available at https://heasarc.gsfc.nasa.gov/w3browse/star-catalog/sao.html
READ_SAO.py is intended to read the file available there.
