### read SAO star catalogue ###
import numpy as np
import datetime
import os

##;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
##;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
##;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
##;;;;;;;;;;;;   precess and premat is included from ASTROLIB without changes (IDL original)
##;;;;;;;;;;;; Translated from fairly faithfully from IDL to Python.

def precess(ra,dec,equinox1,equinox2,FK4 = False,RADIAN = False):
    
##;+
##; NAME:
##;      PRECESS
##; PURPOSE:
##;      Precess coordinates from EQUINOX1 to EQUINOX2.  
##; EXPLANATION:
##;      For interactive display, one can use the procedure ASTRO which calls 
##;      PRECESS or use the /PRINT keyword.   The default (RA,DEC) system is 
##;      FK5 based on epoch J2000.0 but FK4 based on B1950.0 is available via 
##;      the /FK4 keyword.
##;
##;      Use BPRECESS and JPRECESS to convert between FK4 and FK5 systems
##; CALLING SEQUENCE:
##;      PRECESS, ra, dec, [ equinox1, equinox2, /PRINT, /FK4, /RADIAN ]
##;
##; INPUT - OUTPUT:
##;      RA - Input right ascension (scalar or vector) in DEGREES, unless the 
##;              /RADIAN keyword is set
##;      DEC - Input declination in DEGREES (scalar or vector), unless the 
##;              /RADIAN keyword is set
##;              
##;      The input RA and DEC are modified by PRECESS to give the 
##;      values after precession.
##;
##; OPTIONAL INPUTS:
##;      EQUINOX1 - Original equinox of coordinates, numeric scalar.  If 
##;               omitted, then PRECESS will query for EQUINOX1 and EQUINOX2.
##;      EQUINOX2 - Equinox of precessed coordinates.
##;
##; OPTIONAL INPUT KEYWORDS:
##;      /PRINT - If this keyword is set and non-zero, then the precessed
##;               coordinates are displayed at the terminal.    Cannot be used
##;               with the /RADIAN keyword
##;      /FK4   - If this keyword is set and non-zero, the FK4 (B1950.0) system
##;               will be used otherwise FK5 (J2000.0) will be used instead.
##;      /RADIAN - If this keyword is set and non-zero, then the input and 
##;               output RA and DEC vectors are in radians rather than degrees
##;
##; RESTRICTIONS:
##;       Accuracy of precession decreases for declination values near 90 
##;       degrees.  PRECESS should not be used more than 2.5 centuries from
##;       2000 on the FK5 system (1950.0 on the FK4 system).
##;
##; EXAMPLES:
##;       (1) The Pole Star has J2000.0 coordinates (2h, 31m, 46.3s, 
##;               89d 15' 50.6"); compute its coordinates at J1985.0
##;
##;       IDL> precess, ten(2,31,46.3)*15, ten(89,15,50.6), 2000, 1985, /PRINT
##;
##;               ====> 2h 16m 22.73s, 89d 11' 47.3"
##;
##;       (2) Precess the B1950 coordinates of Eps Ind (RA = 21h 59m,33.053s,
##;       DEC = (-56d, 59', 33.053") to equinox B1975.
##;
##;       IDL> ra = ten(21, 59, 33.053)*15
##;       IDL> dec = ten(-56, 59, 33.053)
##;       IDL> precess, ra, dec ,1950, 1975, /fk4
##;
##; PROCEDURE:
##;       Algorithm from Computational Spherical Astronomy by Taff (1983), 
##;       p. 24. (FK4). FK5 constants from "Astronomical Almanac Explanatory
##;       Supplement 1992, page 104 Table 3.211.1.
##;
##; PROCEDURE CALLED:
##;       Function PREMAT - computes precession matrix 
##;
##; REVISION HISTORY
##;       Written, Wayne Landsman, STI Corporation  August 1986
##;       Correct negative output RA values   February 1989
##;       Added /PRINT keyword      W. Landsman   November, 1991
##;       Provided FK5 (J2000.0)  I. Freedman   January 1994
##;       Precession Matrix computation now in PREMAT   W. Landsman June 1994
##;       Added /RADIAN keyword                         W. Landsman June 1997
##;       Converted to IDL V5.0   W. Landsman   September 1997
##;       Correct negative output RA values when /RADIAN used    March 1999 
##;       Work for arrays, not just vectors  W. Landsman    September 2003 
##;-     
##  On_error,2                                           ;Return to caller
##  npar = N_params()
    deg_to_rad = np.pi/180.0
##    array_size = ra.ndims
    npts = len(ra)
    assert len(ra)==len(dec)
    #array  = size(ra,/N_dimen) GE 2
##    if array_size >=2:
##        dimen = np.shape(ra)
    
    #if array then dimen = size(ra,/dimen)
    if RADIAN == False:
        ra_rad = ra*deg_to_rad     
        dec_rad = dec*deg_to_rad 
    else:
        ra_rad= ra
        dec_rad = dec
  
    a = np.cos( dec_rad )  
 
    x = np.zeros((3,npts))
    x[0] = a*np.cos(ra_rad)
    x[1] = a*np.sin(ra_rad)
    x[2] = np.sin(dec_rad)
    x = np.transpose(x)
    
   
    sec_to_rad = deg_to_rad/3600
## Use PREMAT function to get precession matrix from Equinox1 to Equinox2
    r = premat(equinox1, equinox2, FK4 = FK4)
    x2 = np.matmul(x,r)      #rotate to get output direction cosines
    print(np.shape(x2))
    if npts == 1:
        print('scalar')
        ra_rad = np.arctan2(x2[0,1],x2[0,0])
        dec_rad = np.arcsin(x2[0,2])
    else:
        ra_rad = np.zeros(npts) + np.arctan2(x2[:,1],x2[:,0])
        dec_rad = np.zeros(npts) + np.arcsin(x2[:,2])
    if RADIAN == False:
        ra = ra_rad/deg_to_rad
        temp = np.where(ra<0,1,0)
        ra = ra + temp*360.0          #;RA between 0 and 360 degrees
        dec = dec_rad/deg_to_rad
    else:
        ra = ra_rad
        dec = dec_rad
        temp = np.where(ra<0,1,0)
        ra = ra + temp*2.0*np.pi
  
    return ra, dec

def premat(equinox1,equinox2,FK4 = False):
##;+
##; NAME:
##;       PREMAT
##; PURPOSE:
##;       Return the precession matrix needed to go from EQUINOX1 to EQUINOX2.  
##; EXPLANTION:
##;       This matrix is used by the procedures PRECESS and BARYVEL to precess 
##;       astronomical coordinates
##;
##; CALLING SEQUENCE:
##;       matrix = PREMAT( equinox1, equinox2, [ /FK4 ] )
##;
##; INPUTS:
##;       EQUINOX1 - Original equinox of coordinates, numeric scalar.  
##;       EQUINOX2 - Equinox of precessed coordinates.
##;
##; OUTPUT:
##;      matrix - double precision 3 x 3 precession matrix, used to precess
##;               equatorial rectangular coordinates
##;
##; OPTIONAL INPUT KEYWORDS:
##;       /FK4   - If this keyword is set, the FK4 (B1950.0) system precession
##;               angles are used to compute the precession matrix.   The 
##;               default is to use FK5 (J2000.0) precession angles
##;
##; EXAMPLES:
##;       Return the precession matrix from 1950.0 to 1975.0 in the FK4 system
##;
##;       IDL> matrix = PREMAT( 1950.0, 1975.0, /FK4)
##;
##; PROCEDURE:
##;       FK4 constants from "Computational Spherical Astronomy" by Taff (1983), 
##;       p. 24. (FK4). FK5 constants from "Astronomical Almanac Explanatory
##;       Supplement 1992, page 104 Table 3.211.1.
##;
##; REVISION HISTORY
##;       Written, Wayne Landsman, HSTX Corporation, June 1994
##;       Converted to IDL V5.0   W. Landsman   September 1997
##;-    
##  On_error,2                                           #Return to caller
##  npar = N_params()
##   if ( npar LT 2 ) then begin 
##     print,'Syntax - PREMAT, equinox1, equinox2, /FK4]'
##     return,-1 
##  endif 
    deg_to_rad = np.pi/180
    sec_to_rad = deg_to_rad/3600
    t = 0.001*( equinox2 - equinox1)
    if FK4 == False:
        st = 0.001*( equinox1 - 2000)
##;  Compute 3 rotation angle
        A = sec_to_rad * t * (23062.181 + st*(139.656 +0.0139*st) + t*(30.188 - 0.344*st+17.998*t))
        B = sec_to_rad * t * t * (79.280 + 0.410*st + 0.205*t) + A
        C = sec_to_rad * t * (20043.109 - st*(85.33 + 0.217*st) + t*(-42.665 - 0.217*st -41.833*t))
    else: 
        st = 0.001*( equinox1 - 1900)
#  Compute 3 rotation angles
        A = sec_to_rad * t * (23042.53 + st*(139.75 +0.06*st) + t*(30.23 - 0.27*st+18.0*t))
        B = sec_to_rad * t * t * (79.27 + 0.66*st + 0.32*t) + A
        C = sec_to_rad * t * (20046.85 - st*(85.33 + 0.37*st) + t*(-42.67 - 0.37*st -41.8*t))

    sina = np.sin(A) 
    sinb = np.sin(B)
    sinc = np.sin(C)
    cosa = np.cos(A)
    cosb = np.cos(B)
    cosc = np.cos(C)
    r = np.zeros((3,3))
    r[0] = np.array([ cosa*cosb*cosc-sina*sinb, sina*cosb+cosa*sinb*cosc,  cosa*sinc])
    r[1] = np.array([-cosa*sinb-sina*cosb*cosc, cosa*cosb-sina*sinb*cosc, -sina*sinc])
    r[2] = np.array([-cosb*sinc, -sinb*sinc, cosc])
    return r
  

def rd_ae(ra,dec,mjs,lat,lon,deg_input = True):
    #  FAIRLY FAITHFUL TRANSLATION FROM DAN WHITER'S IDL CODE
    # a routine to convert right ascension and declination to azimuth and elevation
    #
    # Inputs:
    #  ra, dec - Right ascension and declination
    #  mjs - Time in mjs
    #  lat, lon - Latitude and longitude of observer (i.e. camera), in degrees.
    # Outputs:
    #  az, el - Azimuth and elevation, in degrees
    # If keyword rad is set, az and el are returned in radians, and ra and dec
    # should be specified in radians.
    #
    gmst=mjs2gmst(mjs)
    lst = gmst+lon/15.
    lst=lst*2.*np.pi/24.
    radeg = 57.295776367187507
    # converting to radians
    
    tor = 0.017453292384744020
    rlat=lat*tor
    if deg_input==True:
    # converting to radians
        rra=ra*15.*tor
        rdec=dec*tor
    else: 
        rra=ra
        rdec=dec

    ha=lst-rra
    # working out elevation
    rel=np.arcsin( np.sin(rdec)*np.sin(rlat)+np.cos(rdec)*np.cos(ha)*np.cos(rlat))
    if deg_input==True:
        el=rel*radeg
    else:
        el=rel
    # working out azimuth
    az=np.arctan2((-np.cos(rdec)*np.sin(ha)),(np.sin(rdec)*np.cos(rlat)-np.cos(rdec)*np.cos(ha)*np.sin(rlat)))
    #print(az)
    if deg_input==True:
        az=az*radeg 
    return az,el

def mjs2gmst(mjs):
#
# procedure to get Greenwich mean sidereal time
#    See Astronomical Algorithms
#       by Jean Meeus, p. 84 (Eq. 11-4) for the constants used.
# Input is mjs, mean jullian seconds.
# Output is gmst.
#

    c = [280.46061837, 360.98564736629, 0.000387933, 38710000.0 ]
    mjs2000 = tt_mjs(2000,1,1,12,0,0,0.0)
    t0 = (mjs-mjs2000)/86400
    t = t0/36525
#
#                            Compute GST in seconds.
#
    gmst = c[0] + (c[1] * t0) + t**2*(c[2] - t/ c[3] )
#
#
    gmst=(gmst/15.0) % 24.0
    s=np.where(gmst < 0)[0]
    count = len(s)
    
    if (count > 0):
        gmst[s]=24.0+gmst[s]

    return gmst

def tt_mjs(year, month, day, hour, minute, sec, millisec):
# FAIRLY FAITHFUL TRANSLATION FROM DAN WHITER'S IDL CODE
#  converts calendar date to modified Julian second
#   (seconds elapsed since 00:00 UT on Jan 1, 1950)
#  
#
    mjs = 0.0
    jj = 0
    l = 0
    jj =int((14-month)/12) # long(month le 2)
    l = year - int(1900*(int(year/1900))) - jj
    mjs = day - 18234 + int((1461*l)/4) + int((367*(month-2+jj*12))/12)
    mjs = mjs * 86400.0
    mjs = mjs + hour*3600 + minute*60 +sec + millisec/1000
    return mjs
"""
Byte-by-byte Description of file: sao.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1-  6  I6     ---     SAO      [1/258997]+ SAO Catalog number
       7  A1     ---     delFlag  [D] if star deleted (ignore all fields)
   8-  9  I2     h       RAh      Hours RA, Equinox=B1950, Epoch=1950.0
  10- 11  I2     min     RAm      Minutes RA, equinox B1950, Epoch=1950.0
  12- 17  F6.3   s       RAs      Seconds RA, equinox B1950, Epoch=1950.0
  18- 24  F7.4   s/a     pmRA     Annual proper motion in RA (FK4 system)
  25- 26  I2     mas/a   e_pmRA   Standard deviation in pmRA
      27  A1     ---     RA2mFlag [+-] '+', add 1, or '-', substract 1,
                                    RA minute: indication that the minutes
                                    of time associated with the seconds
                                    portion RA2 must be increased or
                                    decreased by 1
  28- 33  F6.3   s       RA2s     Seconds portion of RA, original epoch,
                                    precessed to B1950
  34- 35  I2     10mas   e_RA2    Standard deviation of RA2
  36- 41  F6.1   a       EpRA2    Epoch of RA2 (RA original epoch)
      42  A1     ---     DE-      Sign Dec, equinox B1950, Epoch=1950.0
  43- 44  I2     deg     DEd      Degrees Dec, equinox B1950, Epoch=1950.0
  45- 46  I2     arcmin  DEm      Minutes Dec, equinox B1950, Epoch=1950.0
  47- 51  F5.2   arcsec  DEs      Seconds Dec, equinox B1950, Epoch=1950.0
  52- 57  F6.3  arcsec/a pmDE     ? Annual proper motion in Dec (FK4 system) (9)
  58- 59  I2     mas/a   e_pmDE   Standard deviation of Dec proper motion
      60  A1     ---     D2m_Flag [+-] '+', add 1, or '-', substract 1:
                                    Indication that the arcminutes
                                    associated with DE2 must be increased or
                                    decreased by 1
  61- 65  F5.2   arcsec  DE2s     Seconds of Declination, original epoch,
                                    precessed to B1950
  66- 67  I2     10mas   e_DE2    Standard deviation of DE2
  68- 73  F6.1   a       EpDE2    Epoch of DE2 (Declinaation original epoch)
  74- 76  I3     10mas   e_Pos    Standard deviation of position at epoch 1950.0
  77- 80  F4.1   mag     Pmag     []?=99.9 Photographic magnitude
  81- 84  F4.1   mag     Vmag     []?=99.9 Visual magnitude
  85- 87  A3     ---     SpType   Spectral type, '+++' for composite spectra
  88- 89  I2     ---     r_Vmag   Coded source of visual magnitude (1)
  90- 91  I2     ---     r_Num    Coded source of star number and footnotes (2)
      92  I1     ---     r_Pmag   Coded source of photographic magnitude (3)
      93  I1     ---     r_pmRA   Coded source of proper motions (4)
      94  I1     ---     r_SpType Coded source of spectral type (5)
      95  I1     ---     Rem      Coded remarks duplicity and variability (6)
      96  I1     ---     a_Vmag   Accuracy of V: 0 = 2 decimals, 1=1 decimal
      97  I1     ---     a_Pmag   Accuracy of Ptg: 0 = 2 decimals, 1=1 decimal
  98- 99  I2     ---     r_Cat    Code for source catalog (7)
 100-104  I5     ---     CatNum   Number in source catalog
 105-117  A13    ---     DM       Durchmusterung identification, as
                                    catalog 'BD', 'CD', 'CP' (A2)
                                    zone and number (A8), and
                                    component identification (A2) if there
                                        are two or more SAO stars having
                                        the same DM number
                                    supplement letter (A1) for BD
                                        (Warren and Kress, Catalogue /)
 118-123  A6     ---     HD       Henry Draper Catalog (HD or HDE) number (A6)
                                    (Catalogue )
     124  A1     ---     m_HD     HD component and multiple code (8)
 125-129  A5     ---     GC       Boss General Catalog (GC) number
                                    (Catalogue /)
 130-139  D10.8  rad     RArad    Right ascension, 1950.0, in radians
 140-150  D11.8  rad     DErad    Declination, 1950.0, in radians
 151-152  I2     h       RA2000h  Hours RA, equinox, epoch J2000.0
 153-154  I2     min     RA2000m  Minutes RA, equinox, epoch J2000.0
 155-160  F6.3   s       RA2000s  Seconds RA, equinox, epoch J2000.0
 161-167  F7.4   s/a     pmRA2000 Annual proper motion FK5 system
     168  A1     ---     DE2000-  Sign Dec, equinox, epoch J2000.0
 169-170  I2     deg     DE2000d  Degrees Dec, equinox, epoch J2000.0
 171-172  I2     arcmin  DE2000m  Minutes Dec, equinox, epoch J2000.0
 173-177  F5.2   arcsec  DE2000s  Seconds Dec, equinox, epoch J2000.0
 178-183  F6.3  arcsec/a pmDE2000 ? Annual proper motion in FK5 system (9)
 184-193  D10.8  rad    RA2000rad Right ascension, J2000.0, in radians
 194-204  D11.8  rad    DE2000rad Declination, J2000.0, in radians

"""


def read_sao(directory,year = None, month = None, day = None, hour = None, minute = None,second = None,GET_VMAG = False):
    if ((year!=None) and (month!=None) and (day!=None) and (hour!=None) \
        and (minute!=None) and (second!=None)):
        PROCESS = True
    else:
        PROCESS = False
        
    file = os.path.join(directory,'sao')

    f = open(file,'r')
    F = f.readlines()[:]
    f.close()

    SAO = []
    HD = []
    RA = []
    DEC = []
    VMAG = []
    row = 0
    for line in F:
        
        SAO.append(int(line[0:6]))
        try:
            HD.append(int(line[117:123]))
        except:
            HD.append(np.nan)
        RA.append(float(line[183:193]))
        DEC.append(float(line[193:204]))
        VMAG.append(float(line[80:84]))
        row+=1


    SAO = np.array(SAO)
    HD = np.array(HD)
    RA = np.array(RA)
    DEC = np.array(DEC)
    VMAG = np.array(VMAG)

    if PROCESS == False:
        print('returning J2000 (i.e., not precessed) ascension and declination in radians')
        if GET_VMAG:
            return SAO,HD,RA,DEC,VMAG
        else:
            return SAO,HD,RA,DEC

    else:

        date = [year,month,day,hour,minute,second]
        if date[0]%4 == 0:
            if date[0]%400==0:
                dsoy = 366
            elif date[0]%100 == 0:
                dsoy = 365
            else:
                dsoy = 366
        else:
            dsoy = 365
        mjs = tt_mjs(date[0],date[1],date[2],date[3],date[4],date[5],0)
        doy = datetime.datetime(date[0],date[1],date[2],date[3],date[4],date[5],0).timetuple().tm_yday-1
        ra0,dec0 = precess(np.array(RA),np.array(DEC),2000,date[0]+(doy+(date[3]+date[4]/60)/24)/dsoy,RADIAN = True)
        print('returning precessed ascension and declination in radians for ',str(year)+'/'+str(month).zfill(2)+'/'+\
              str(day).zfill(2)+' '+str(hour).zfill(2)+':'+str(minute).zfill(2)+':'+str(second).zfill(2))
        if GET_VMAG:
            return SAO,HD,ra0,dec0,VMAG
        else:
            return SAO,HD,ra0,dec0
