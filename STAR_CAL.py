import numpy as np
from matplotlib import pyplot as plt
from READ_SAO import read_sao,tt_mjs,rd_ae
from scipy.optimize import differential_evolution
from scipy.signal import convolve2d
from matplotlib.widgets import TextBox
import glob
import os
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def geodet2cen(lat,lon,alt):
    X,Y,Z = geodet2cart(lat,alt,lon)#,a2 = None,b2 = None)
    lat, lon, r = cart2geocen(X,Y,Z)
    return lat,lon,r

def cart2geocen(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    
    return lat, lon, r

def geodet2cart(detlat,detalt,lon):#,a2 = None,b2 = None):
    #if a2==None:
    a2=40680631.59 #semi-major axis squared in km**2
    #if b2 == None:
    b2=40408299.98 #semi-minor axis squared in km**2


    rlon = np.radians(lon)
    rdetlat = np.radians(detlat)

    N = a2/(np.sqrt(a2*np.cos(rdetlat)**2 + b2 * np.sin(rdetlat)**2))
    X = (N+detalt) * np.cos(rdetlat)*np.cos(rlon)
    Y = (N+detalt) * np.cos(rdetlat)*np.sin(rlon)
    Z = (b2/a2 * N + detalt) * np.sin(rdetlat)
    return X,Y,Z
def angular_difference(a,b):
    ## gets round the angle wrapping issue (-179 and 179 are really close angles).
    return np.arctan2(np.sin(a-b), np.cos(a-b))
def gaussian_kernel(L=15, sigma=15/4):
    axis = np.linspace(-(L-1)/2.0,(L-1)/2.0,L)
    gauss = np.exp(-0.5 * np.square(axis) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def detrend_image(raw_im_file,L,SIG,Rlim = 2500,Blim = 10,IMTYPE = 'SCMOS',FROM_FILE = True):
    if FROM_FILE:
        test_im = plt.imread(raw_im_file)
        if IMTYPE == 'ASC':
            test_im = rgb2gray(test_im)
    else:
        test_im = raw_im_file
    dims = np.shape(test_im)
    yvi,xvi = np.meshgrid(np.arange(dims[0]),np.arange(dims[1]),indexing= 'ij')
    cen_x = dims[1]//2
    cen_y = dims[0]//2
    R = np.sqrt((yvi-cen_y)**2 + (xvi-cen_x)**2)

    convim = convolve2d(test_im,gaussian_kernel(L = L,sigma = SIG),mode = 'same')
    detrend = test_im-convim
    detrend = np.where(R>Rlim,0,detrend)
    detrend = np.where(detrend<=Blim,0,detrend)
    return detrend

def get_xy(popt,az,el,TYPE = 'orthographic'):
    #N = np.cos(az)*np.cos(el)
    #E = np.sin(az)*np.cos(el)
    #U = np.sin(el)

    oax = popt[0]
    oay = popt[1]
    alpha = popt[2]
    beta = popt[3]
    gamma = popt[4]
    cam_params = popt[5:]

    ENU = np.array([np.cos(el)*np.sin(az),
                    np.cos(el)*np.cos(az),
                    np.sin(el)])

    yaw = np.array([[np.cos(alpha),-np.sin(alpha),0],
                   [np.sin(alpha),np.cos(alpha),0],
                   [0,0,1]])
    pitch = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    roll = np.array([[1,0,0],
                     [0,np.cos(gamma),-np.sin(gamma)],
                     [0,np.sin(gamma),np.cos(gamma)]])

    basis_cam = roll.T@(pitch.T@(yaw.T@ENU))
    theta = np.arccos(np.clip(basis_cam[2],-1,1))
    phi = np.arctan2(basis_cam[1],basis_cam[0])### arctan(y/x), y in camera is not North.
    ###phi is measured from the +ve x-axis, increasing towards positive y-axis
    if TYPE == 'rectilinear':
        r = cam_params[0]*np.tan(theta)
    elif TYPE == 'stereographic':
        r = cam_params[0]*np.tan(theta/cam_params[1])
    elif TYPE == 'equidistant':
        r = cam_params[0]*theta
    elif TYPE == 'equisolid_angle':
        r = cam_params[0]*np.sin(theta/cam_params[1])
    elif TYPE == 'orthographic':
        r = cam_params[0]*np.sin(theta)
    else:
        raise ValueError('specify lens type')

    x = oax+r*np.cos(phi)
    y = oay+r*np.sin(phi)
    return x,y

def get_azel(popt,x,y,TYPE = None):
    oax = popt[0]
    oay = popt[1]
    alpha = popt[2]
    beta = popt[3]
    gamma = popt[4]
    cam_params = popt[5:]
    from_centre = np.sqrt((x-oax)**2 + (y-oay)**2) ## distance from optical axis
    phi = np.arctan2(y-oay,x-oax)
    #print(np.shape(phi))
    #print(np.shape(from_centre))
    #theta = bounded_theta(poly,from_centre)
    ## theta is the incidence angle, i.e., the angle from the optical axis
    ERRFLAG = 0
    if TYPE == 'rectilinear':
        theta = np.arctan(from_centre/cam_params[0])
    elif TYPE == 'stereographic':
        theta = cam_params[1]*np.arctan(from_centre/cam_params[0])
    elif TYPE == 'equidistant':
        theta = from_centre/cam_params[0]
    elif TYPE == 'equisolid_angle':
        if cam_params[0]<np.max(from_centre):
            cam_params[0] = np.max(from_centre)
            ERRFLAG = 1
        theta = cam_params[1]*np.arcsin(from_centre/cam_params[0])
    elif TYPE == 'orthographic':
        if cam_params[0]<np.max(from_centre):
            cam_params[0] = np.max(from_centre)
            ERRFLAG = 1
        theta = np.arcsin(from_centre/cam_params[0])
    else:
        raise ValueError('specify lens type')

    
    basis_cam = np.array([np.sin(theta)*np.cos(phi),
                      np.sin(theta)*np.sin(phi),
                      np.cos(theta)])
    yaw = np.array([[np.cos(alpha),-np.sin(alpha),0],
                   [np.sin(alpha),np.cos(alpha),0],
                   [0,0,1]])
    pitch = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    roll = np.array([[1,0,0],
                     [0,np.cos(gamma),-np.sin(gamma)],
                     [0,np.sin(gamma),np.cos(gamma)]])
    
    
    ENU = yaw@(pitch@(roll@basis_cam))
    #print(np.shape(ENU))
    el = np.arctan2(ENU[2], np.sqrt(ENU[0]**2 + ENU[1]**2))
    az = np.arctan2(ENU[0], ENU[1])

    return az,el,ERRFLAG

def minimiser(popt,star_x,star_y,ref_az,ref_el,TYPE):

    az,el,ERRFLAG = get_azel(popt,star_x,star_y,TYPE = TYPE)

    weight_az = 1#np.cos(ref_el)**2 ## downweight azimuth errors near zenith (they vanish at the zenith)

    delta_az = np.degrees(angular_difference(az,ref_az))

    resid = np.sum((np.degrees(el)-np.degrees(ref_el))**2 + weight_az*(delta_az)**2)
    return resid+ERRFLAG*1e10
    

def select_stars(cam_geodet_lat,cam_geolon,cam_alt,detrend_im,year,month,day,hour,minute,second,sao_dir,elevation_lim = 20,\
                 how_many = 1000,save_clicks = False,FIND = True):
    '''
    cam_geodet_lat: camera geodetic latitude (deg)
    cam_geolon: camera longitude (deg)
    cam_alt: camera altitude (km)
    detrend_im: 2D array, the detrended or enhanced image. Use detrend_image or use your own.
    year,month,day,hour,minute,second: all int, the timestamp of the image. Needed to correctly precess the SAO stars.
    sao_dir: directory where the SAO catalogue is stored.
    elevation_lim: float, exclude SAO stars with elevation below this
    how_many: int, how many SAO stars to display and how many to find (you can terminate this early. 50 clicked stars is plenty, but when how_many is high it makes it easier
                    to identify constellations.
    save_clicks: bool, whether to save clicked stars. Currently filename is hard coded.
    FIND: bool, whether to get stars by clicking on the image. If False, will try to load previous clicks.
    '''
    global vmax,rotation,counter_star,stop,dims
    SAO,HD,ra0,dec0,VMAG = read_sao(sao_dir,year,month,day,hour,minute,second,GET_VMAG = True)
    dims = np.shape(detrend_im)[0]
    camera_geocen_lat,_,_ =geodet2cen(cam_geodet_lat,cam_geolon,cam_alt)

    mjs = tt_mjs(year,month,day,hour,minute,second,0)
    AZ = np.zeros(len(ra0))
    EL = np.zeros(len(ra0))
    VM = np.zeros(len(ra0))

    mag_ref = 6
    val_ref = 10
    counter = 0
    for i in range(len(ra0)):
        az,el = rd_ae(ra0[i],dec0[i],mjs,camera_geocen_lat,cam_geolon,deg_input = False)
        mag = val_ref * 10**(0.4 * (mag_ref-VMAG[i]))
        if el>=np.radians(elevation_lim[0]) and el<np.radians(elevation_lim[1]):
            AZ[counter] = az
            EL[counter] = el
            VM[counter] = mag
            counter+=1
    AZ = AZ[:counter]
    EL = EL[:counter]
    VM = VM[:counter]

    sor = np.argsort(-VM)
    VM = VM[sor[:how_many]]
    AZ = AZ[sor[:how_many]]
    EL = EL[sor[:how_many]]
    SAO = SAO[sor[:how_many]]

    clicked_x = np.ones_like(AZ)*np.nan
    clicked_y = np.ones_like(AZ)*np.nan
    counter_star = 0

    if FIND:
        ### adjust vmax and rotation of star map so it's easier to identify the stars
        fig = plt.figure(figsize=(13,7))
        

        ax = fig.add_subplot(121, projection='polar')
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        ax.set_facecolor('k')
        

        ax2 = fig.add_subplot(122)
        im = ax2.imshow(detrend_im,cmap = 'magma')
        def on_click_setrot(event):
            if event.inaxes is None:
                print('rotation is now fixed')
                plt.close(fig)
        def set_rot(text):
            global rotation

            try:
                rotation = float(text)
            except ValueError:
                print('must be numerical angle in degrees')
                return
            ax.set_theta_zero_location('N', offset=rotation)
            fig.canvas.draw_idle()
        def set_vmax(text):
            global vmax
            try:
                vmax = float(text)
            except ValueError:
                return

            vmin, _ = im.get_clim()
            im.set_clim(vmin=vmin, vmax=vmax)
            fig.canvas.draw_idle()
        
        vmin, vmax = im.get_clim()

        ax.scatter(AZ,90-np.degrees(EL),s = (15*VM/np.max(VM)*72.0/fig.dpi)**2,c = 'w')
        for i in range(5):
            ax.text(AZ[i],90-np.degrees(EL[i]),str(SAO[i]),color = 'y')
        ax.set_title('Set vmax and rotation so the stars are visible and the star map is orientated conveniently.\nClick outside the axes when you are done')
        ax.grid(False)
        rotation = 0
        rot_ax = fig.add_axes([0.25, 0.02, 0.12, 0.05])
        rot_box = TextBox(rot_ax, "Rotation", initial="0")
        vmax_ax = fig.add_axes([0.45, 0.02, 0.12, 0.05])
        vmax_box = TextBox(vmax_ax, "vmax", initial=str(int(im.get_clim()[1])))
        rot_box.on_submit(set_rot)
        vmax_box.on_submit(set_vmax)

        fig.canvas.mpl_connect('button_press_event', on_click_setrot)

        plt.show()
        stop = False
        for i in range(len(AZ)):
            if stop:
                break
            fig = plt.figure(figsize = (13,7))
            def onclick(event):
                global counter_star,dims
                # only accept clicks inside axes
                if event.inaxes is None:
                    return

                x, y = event.xdata, event.ydata
                
                if x<dims//20 and y<dims//20:
                    x = np.nan
                    y = np.nan
                    print('skipping')
                else:
                    print(f'clicked a star at: x={x:.3f}, y={y:.3f}')
                

                clicked_x[counter_star] = x
                clicked_y[counter_star] = y
                counter_star +=1

                # close figure to unblock loop
                plt.close()
            def on_click_end(event):
                global stop
                if event.inaxes is None:
                    print('closing')
                    stop = True
                    plt.close(fig)
            
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            cid2 = fig.canvas.mpl_connect('button_press_event', on_click_end)
            ax = fig.add_subplot(121,projection = 'polar')
            
            ax2 = fig.add_subplot(122)
            ax.scatter(AZ,90-np.degrees(EL),s = (15*VM/np.max(VM)*72.0/fig.dpi)**2,c = 'w')
            ax.set_facecolor('k')
            ax.set_title(f'{i} of {len(AZ)-1}')
            ax.scatter(AZ[i],90-np.degrees(EL[i]),edgecolors = 'r',facecolors = 'none',marker = 'o')
            ax.set_theta_zero_location('N',offset=rotation)
            ax.set_theta_direction(1)
            ax.grid(False)
            ax2.imshow(detrend_im,cmap = 'magma',vmin = vmin,vmax = vmax)
            ax2.plot(np.ones(dims//20)*dims//20,np.arange(dims//20),color = 'r')
            ax2.plot(np.arange(dims//20),np.ones(dims//20)*dims//20,color = 'r')
            ax2.set_title('Click on the star in this axis that is highlighted on the polar plot.\nClick inside the red box to skip a star.\nClick outside the axes to end.')
            plt.show()
        clicked_x = clicked_x[:counter_star]
        clicked_y = clicked_y[:counter_star]
        if save_clicks:
            f = open(os.path.join(sao_dir,'clicks.txt'),'w')
            for j in range(len(clicked_x)):
                f.write(str(SAO[j])+' '+str(clicked_x[j])+' '+str(clicked_y[j])+'\n')
            f.close()
    else:
        f = open(os.path.join(sao_dir,'clicks.txt'),'r')
        F = f.readlines()[:]
        f.close()
        row = 0
        for line in F:
            col = line.split()
            clicked_x[row] = float(col[1])
            clicked_y[row] = float(col[2])
            row+=1
        counter_star = row
    valid = np.where(np.isfinite(clicked_x))[0]
    
    return clicked_x,clicked_y,valid,AZ,EL,VM,SAO


def fit(detrend_im,clicked_x,clicked_y,valid,AZ,EL,VM,SAO,PROJECTION = 'orthographic',x0 = None,init = 'sobol',popsize = 500,bounds = None):
    
    ### you might need to play around with the fitting popsize etc to get a good fit
    dims = np.shape(detrend_im)
    if not bounds:
        bounds = [(-dims[1],2*dims[1]),\
                  (-dims[0],2*dims[0]),\
                  (-np.pi/2,np.pi/2),\
                  (-np.pi,np.pi),\
                  (-np.pi,np.pi),\
                  (dims[1]/8,dims[1]*2)]
        
    
    args = (clicked_x[valid],clicked_y[valid],AZ[valid],EL[valid],PROJECTION)
    res = differential_evolution(minimiser,x0 = x0,args = args,init = init,tol = 1e-8,bounds = bounds,disp = True,popsize = popsize)
    popt = res.x

    star_az,star_el,_ =get_azel(popt,clicked_x[valid],clicked_y[valid],TYPE = PROJECTION)

    ax = plt.subplot(111,projection = 'polar')
    ax.scatter(AZ[valid],90-np.degrees(EL[valid]),marker = 'o',facecolors = 'none',edgecolors = 'blue')
    ax.scatter(star_az,90-np.degrees(star_el),marker = '+',c = 'r')
    ax.set_theta_zero_location('N')
    ax.set
    plt.show()
    return popt


def composite_image(base_dir):
    files = glob.glob(os.path.join(base_dir,'*.tif'))
    for i in range(len(files)):
        if i ==0:
            im = plt.imread(files[i])
        else:
            im+=plt.imread(files[i])
    return im

def get_bounds(popt):
    bounds = []
    for i in range(len(popt)):
        if popt[i]>0:
            bounds.append((popt[i]*0.5,popt[i]*1.5))
        else:
            bounds.append((popt[i]*1.5,popt[i]*0.5))
    return bounds

def geo_zenith(popt,TYPE = None):
    oax = popt[0]
    oay = popt[1]
    alpha = popt[2]
    beta = popt[3]
    gamma = popt[4]
    cam_params = popt[5:]
    yaw = np.array([[np.cos(alpha),-np.sin(alpha),0],
                   [np.sin(alpha),np.cos(alpha),0],
                   [0,0,1]])
    pitch = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    roll = np.array([[1,0,0],
                     [0,np.cos(gamma),-np.sin(gamma)],
                     [0,np.sin(gamma),np.cos(gamma)]])

    zen = np.array([0,0,1])

    zen_cam_basis = roll.T@(pitch.T@(yaw.T@zen))
    theta = np.arccos(np.clip(zen_cam_basis[2],-1,1))
    phi = np.arctan2(zen_cam_basis[1],zen_cam_basis[0])

    if TYPE == 'rectilinear':
        r = cam_params[0]*np.tan(theta)
    elif TYPE == 'stereographic':
        r = cam_params[0]*np.tan(theta/cam_params[1])
    elif TYPE == 'equidistant':
        r = cam_params[0]*theta
    elif TYPE == 'equisolid_angle':
        r = cam_params[0]*np.sin(theta/cam_params[1])
    elif TYPE == 'orthographic':
        r = cam_params[0]*np.sin(theta)
    else:
        raise ValueError('specify lens type')
    x = oax + r * np.cos(phi)
    y = oay + r * np.sin(phi)
    return x,y
        
   
composite = composite_image(r'/path/to/test_ims/')
##detrend_im = detrend_image(r'/users/nick/documents/phd/codes/Sony2016-01-11/LYR-Sony-110116_190014.jpg',Rlim = 2250,Blim = 10,L = 50,SIG = 50/4)

### SCMOS EXAMPLE: this is a good fit: np.array([ 3.49199330e+02,  4.90378864e+02, -2.78998473e+00,  2.94935836e-03, -1.18548044e-01,  3.94774827e+03])
### popt[0] = optical axis x coord; popt[1] = optical axis y coord; popt[2] = rotation1 ; popt[3] = rotation2; popt[4] = rotation3; popt[5] = focal length
### the above is a very good fit, but the optical axis seems a bit far from the centre of the image.
detrend_im = detrend_image(composite,Rlim = 500,Blim = 10,L = 20,SIG = 20/4,IMTYPE = 'SCMOS',FROM_FILE = False)

sao_dir = r'/path/to/sao_dir/'
cam_geodet_lat = 78.14796
cam_geolon = 16.0430002
cam_alt = 0.520
year = 2016
month = 1
day = 11
hour = 19
minute = 0
second = 2
clicked_x,clicked_y,valid,AZ,EL,VM,SAO=select_stars(cam_geodet_lat,cam_geolon,cam_alt,detrend_im,year,month,day,hour,minute,second,sao_dir,\
                                                    elevation_lim = [70,90],how_many = 1000,save_clicks = False,FIND = True)
dims = np.shape(detrend_im)
bounds = [(0,dims[1]),\
                  (0,dims[0]),\
                  (-np.pi,np.pi),\
                  (-np.pi,np.pi),\
                  (-np.pi,np.pi),\
                  (dims[1]/8,dims[1]*4),\
                  (0.5,3)]
popt = fit(detrend_im,clicked_x,clicked_y,valid,AZ,EL,VM,SAO,PROJECTION = 'rectilinear',x0 = None,init = 'sobol',popsize = 100,bounds = bounds)
Nfit = 2
### refining the fit a bit
for _ in range(Nfit):
    popt = fit(detrend_im,clicked_x,clicked_y,valid,AZ,EL,VM,SAO,PROJECTION = 'rectilinear',x0 = popt,init = 'random',popsize = 100,bounds = get_bounds(popt))
    print(popt)
    

    
