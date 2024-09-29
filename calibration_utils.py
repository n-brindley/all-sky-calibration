import numpy as np



#### functions:
#### back2x: finds image x coordinate from given azimuth and elevation
#### back2y: finds image y coordinate from given azimuth and elevation
#### el_func: finds elevation for given image xy coords
#### az_func: finds azimuth for given image xy coords


#### parameters:
#### x0,y0: coordinates of geographic zenith
#### P2: scaling coefficient for calculating elevation; scales the radial distance from the zenith
#### P3: rotation offset for calculating the azimuth
#### P4: another coefficient for calculating elevation;
####     only relevant if it's a fish-eye lens and gives quadratic behaviour towards the edge of the FOV


'''
Narrow Field of view sCMOS camera params for 2016/01/11 19:01:00
                x0            y0                P2              P3             P4
np.arary([3.38512515e+02, 2.62800220e+01, 1.45331577e-02, 1.60068426e+02, 1.00000000e-50])

'''





def back2y(y0,el,az,P2,P3,P4):

    ## el: elevation in degrees
    ## az: azimuth in degrees

    ### because of the sqrt later, we need to determine whether we want the + or - solution
    az_check = az-P3
    az_check = np.where(az_check<-180,az_check+360,az_check)
    az_check = np.where(az_check>180,az_check-360,az_check)
    sgn_x = np.where(az_check<0,-1,1)
    sgn_y = np.where((az_check>=-90)&(az_check<=90),1,-1)

    #### if P4 is very small (i.e., the lens is not fish-eye), we get weird behaviour, probably due to overflow
    #### so the if-else statements get round this
    if P4>1e-10:
        top = np.sqrt((np.sqrt((P2/(2*P4))**2 + (90-el)/P4) - P2/(2*P4))**2)
    else:
        top = np.sqrt(((90-el)/P2)**2)
    bottom = np.sqrt(1 + (np.tan(np.radians(az-P3)))**2)
    
    res = sgn_y*top/bottom + y0
    return res

def back2x(x0,el,az,P2,P3,P4):

    ## el: elevation in degrees
    ## az: azimuth in degrees
    
    ### because of the sqrt later, we need to determine whether we want the + or - solution
    az_check = az-P3
    az_check = np.where(az_check<-180,az_check+360,az_check)
    az_check = np.where(az_check>180,az_check-360,az_check)
    sgn_x = np.where(az_check<0,-1,1)
    sgn_y = np.where((az_check>=-90)&(az_check<=90),1,-1)
    
    #### if P4 is very small (i.e., the lens is not fish-eye), we get weird behaviour, probably due to overflow
    #### so the if-else statements get round this
    if P4>1e-10:
        top = np.sqrt((np.sqrt((P2/(2*P4))**2 + (90-el)/P4) - P2/(2*P4))**2)
    else:
        top = np.sqrt(((90-el)/P2)**2)
    bottom = np.sqrt(1 + 1/((np.tan(np.radians(az-P3)))**2))
    res = sgn_x*top/bottom + x0
    return res

def el_func(x,y,x0,y0,P2,P4):
    ## r is radial distance from zenith in xy coords ##
    r = np.sqrt((x-x0)**2 + (y-y0)**2)
    el = 90-r*P2 - P4*r**2
    return el  ## in degrees

def az_func(x,y,x0,y0,P3):
    ## r is radial distance from zenith in xy coords ##
    r = np.sqrt((x-x0)**2 + (y-y0)**2)
    az =np.degrees(np.arctan2((x-x0),(y-y0)))+P3
    return az  ## in degrees
