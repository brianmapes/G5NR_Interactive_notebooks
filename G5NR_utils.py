import xarray as xr
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib import request

try:
   import holoviews as hv
   import geoviews as gv
   def to_holoimage(data,dynamic=True,style_opts={'cmap':'RdBu_r'},plot_opts={'width':600,'toolbar':'above','colorbar':True}):
       hvd=hv.Dataset(data)
       if len(hvd.dimensions())<4:
          return hvd.to(hv.Image,kdims=['lon','lat'])(plot=plot_opts)(style=style_opts)
       return hvd.to(hv.Image,kdims=['lon','lat'],dynamic=dynamic)(plot=plot_opts)(style=style_opts)
   def to_geoimage(data,dynamic=True,style_opts={'cmap':'RdBu_r'},plot_opts={'width':600,'toolbar':'above','colorbar':True},hover=False):
       gvd=gv.Dataset(data)
       if len(gvd.dimensions())<4:
          gvimg=gvd.to(gv.Image,kdims=['lon','lat'])(plot=plot_opts)(style=style_opts)
       else:
          gvimg=gvd.to(gv.Image,kdims=['lon','lat'],dynamic=dynamic)(plot=plot_opts)(style=style_opts)
       if hover:
          projected = gv.operation.project_image(gvimg)
          gvimg=hv.QuadMesh(projected,kdims=gvimg.kdims,vdims=gvimg.vdims)(plot=plot_opts)(style=style_opts)
          gvimg=gvimg(plot={'tools':['hover']})
          #gvimg*=gvd.to(gv.Points,kdims=['lon','lat'])(style={'alpha':0,'marker':'square','size':6})(plot={'tools':['hover']})
       return gvimg #*gv.feature.coastline
   xr.DataArray.to_holoimage=to_holoimage
   xr.DataArray.to_geoimage=to_geoimage
except Exception as err:
   print('Functionality related to holoviews cannot be setup because:  {0}'.format(err))

try:
   from pyproj import Proj, transform
   inProj = Proj(init='epsg:3857')
   outProj = Proj(init='epsg:4326')
   def merc_dist2lonlat(xdist,ydist):
      return transform(inProj,outProj,xdist,ydist)
   def merc_lonlat2dist(lon,lat):
      return transform(outProj,inProj,lon,lat)
except Exception as err:
   print('Projection transformation related function will be missing because: {0}'.format(err))
def genlon_bins(nlon):
    """mainly to match cdo calculations"""
    """watch out when close to 180"""
    import itertools
    start=-180.0
    stop=180.0
    dy=(stop-start)/float(nlon)
    ra=itertools.count(start-dy/2,dy)
    return [next(ra) for i in range(nlon+1)]
def genlon_bins2(nlon):
    """mainly to match cdo calculations"""
    """watch out when close to 180"""
    import itertools
    dx=360.0/nlon
    start=-180.0-dx/2
    stop=180.0+dx/2
    curr=start
    while curr<stop:
       yield curr
       curr=curr+dx 
def genlat_bins(nlat):
    """mainly to match cdo calculations"""
    import itertools
    start=-90.0
    stop=90.0
    dy=(stop-start)/float(nlat-1)
    ra=itertools.count(start-dy/2,dy)
    return [next(ra) for i in range(nlat+1)]
def subgrid(variable,nlon=None,nlat=None,res=None):
    #assert isinstance(variable,xr.DataArray),"please pass a xarray DataArray not whole dataset"
    if nlon and nlat:
       lat_bins=genlat_bins(nlat)
       lon_bins=genlon_bins(nlon) #not valid for cyclic grids?
    elif res:
       lon_bins=genlon_bins2(nlon)  
    func1=lambda x:x-x.mean(dim=['lat','lon'])
    func2=lambda y:y.groupby_bins('lon',lon_bins).apply(func1)
    return variable.groupby_bins('lat',lat_bins).apply(func2)
def regrid(variable,nlon,nlat,weights=None):
    #assert isinstance(variable,xr.DataArray),"please pass a xarray DataArray not whole dataset"

    lat_bins=genlat_bins(nlat)
    lon_bins=genlon_bins(nlon) #not valid for cyclic grids?
    if not weights is None:
       assert isinstance(weights,xr.DataArray),'please pass a xarray DataArray for the weights'
       assert len(weights.dims)<3,'weights need to have only lat and lon dimension'
       #varname=variable.name 
       wgtname=weights.name
        
       wgts=weights.sel(lat=variable.lat,lon=variable.lon)
       wgts=wgts/wgts.sum()  
       variable_wgtd=variable*wgts
       if isinstance(variable_wgtd,xr.DataArray):
          variable_wgtd.name=variable.name
   
       variable_wgtd=xr.merge([variable_wgtd,wgts]) #,wgts])
       lat_gpd=variable_wgtd.groupby_bins('lat',lat_bins)
       latreg=lat_gpd.mean(dim=['lat'],skipna=True)
       lon_gpd=latreg.groupby_bins('lon',lon_bins)
       regrided=lon_gpd.mean(dim=['lon'],skipna=True)
       if isinstance(variable,xr.DataArray):
           regrided=regrided[variable.name]/regrided[wgtname]
       else:
           regrided=regrided/regrided[wgtname]
    else:
       lat_gpd=variable.groupby_bins('lat',lat_bins)
       latreg=lat_gpd.mean(dim=['lat'])#,skipna=True,keep_attrs=True)
       lon_gpd=latreg.groupby_bins('lon',lon_bins)
       regrided=lon_gpd.mean(dim=['lon']) #,skipna=True,keep_attrs=True)

    regrided=regrided.rename({'lat_bins':'lat'})
    regrided=regrided.rename({'lon_bins':'lon'})
    lats=[]
    for latbin in regrided.lat.values:
        latbounds=[float(lat) for lat in str(latbin).strip(r'(])').split(',')]
        lats.append(np.mean(latbounds))
    regrided['lat']=lats    

    lons=[]
    for lonbin in regrided.lon.values:
        lonbounds=[float(lon) for lon in str(lonbin).strip(r'(])').split(',')]
        lons.append(np.mean(lonbounds))
    regrided['lon']=lons

    return regrided.sel(lat=slice(variable.lat.min(),variable.lat.max()),lon=slice(variable.lon.min(),variable.lon.max()))
 
def load_from_zidv(url=None,outdir=None):
    assert url.startswith('http:'),'only urls supported at this time'
    remote_file=request.urlopen(url)
    with ZipFile(BytesIO(remote_file.read())) as zip_file:
       for contained_file in zip_file.namelist():
           if str(contained_file)=='data_0_3D7km30minuteInst':
                da_3d=xr.open_dataset(zip_file.extract(contained_file,outdir))
           if str(contained_file)=='data_1_inst30mn_2d_met1_Nx':
                da_2d=xr.open_dataset(zip_file.extract(contained_file,outdir))
    return xr.merge([da_3d,da_2d])

def load_05deg_dataset():
    url='http://weather.rsmas.miami.edu/repository/opendap/synth:1142722f-a386-4c17-a4f6-0f685cd19ae3:L0c1TlIvRzVOUi1BdmcxaC0wLjVkZWctVVZXX1VXX1ZXLm5jbWw=/entry.das'
    return xr.open_dataset(url)

def load_4deg_dataset():
    url='http://weather.rsmas.miami.edu/repository/opendap/synth:1142722f-a386-4c17-a4f6-0f685cd19ae3:L0c1TlIvRzVOUi1BdmcxaC00ZGVnLVVWV19VV19WVy5uY21s/entry.das'
    return xr.open_dataset(url)

def load_4deg_skedot_dataset(lonflip=True):
    url='http://weather.rsmas.miami.edu/repository/opendap/synth:1142722f-a386-4c17-a4f6-0f685cd19ae3:L0c1TlIvU0tFZG90X21lcmdlZF85MHg0NS5uYw==/entry.das'
    url_lonflip='http://weather.rsmas.miami.edu/repository/opendap/synth:1142722f-a386-4c17-a4f6-0f685cd19ae3:L0c1TlIvU0tFZG90X21lcmdlZF85MHg0NV9mbGlwLm5j/entry.das'
    if lonflip:
       return xr.open_dataset(url_lonflip)
    else:
       return xr.open_dataset(url) 

def SKEDot(rho,u,v,w,nlon,nlat):
    u_regrid=regrid(u,nlon,nlat)
    v_regrid=regrid(v,nlon,nlat)
    w_regrid=regrid(w,nlon,nlat) 
    rho_regrid=regrid(rho,nlon,nlat) 
    up=subgrid(u,nlon,nlat)
    vp=subgrid(v,nlon,nlat)
    wp=subgrid(w,nlon,nlat)

    upwp=regrid(up*wp,nlon,nlat)
    upwp.name='upwp'

    vpwp=regrid(vp*wp,nlon,nlat)
    vpwp.name='vpwp'

    uw=regrid(u*w,nlon,nlat)
    uw.name='uw'
    vw=regrid(v*w,nlon,nlat)
    vw.name='vw'

        
    rhoupwp=rho*up*wp
    rhovpwp=rho*vp*wp
    
           
    Eddy_Flux_Zon=regrid(rhoupwp,nlon,nlat)
    Eddy_Flux_Mer=regrid(rhovpwp,nlon,nlat)
    Eddy_Flux_Zon.name='Eddy_Flux_Zon'
    Eddy_Flux_Mer.name='Eddy_Flux_Mer'
#make it a dataset for easy function application on all variables 
    Eddy_Flux=xr.merge([Eddy_Flux_Zon,Eddy_Flux_Mer])
    
    dp=u.lev
    dp=dp*100.0
    dp=np.gradient(dp)
    dPbyg=dp/9.8
    dPbyg=xr.DataArray(dPbyg,coords={'lev':u.lev},dims='lev')
    axisint=1 if len(np.shape(Eddy_Flux_Zon))>3 else 0
    Eddy_Flux_Tend=Eddy_Flux.apply(np.gradient,axis=axisint)
    Eddy_Flux_Tend=Eddy_Flux_Tend/dPbyg
   
    Eddy_Tend_Zon=Eddy_Flux_Tend['Eddy_Flux_Zon']
    Eddy_Tend_Zon.name='Eddy_Tend_Zon'

    Eddy_Tend_Mer=Eddy_Flux_Tend['Eddy_Flux_Mer']
    Eddy_Tend_Mer.name='Eddy_Tend_Mer'


    u_baro=(rho_regrid*u_regrid).sum(dim='lev')/rho_regrid.sum(dim='lev')
    u_baro.name='ubaro'
    v_baro=(rho_regrid*v_regrid).sum(dim='lev')/rho_regrid.sum(dim='lev')
    v_baro.name='vbaro'
    

    ushear=u_regrid-u_baro
    ushear.name='ushear'
    vshear=v_regrid-v_baro
    vshear.name='vshear'

    SKE=(ushear*ushear+vshear*vshear)*0.5
    SKE=(rho_regrid*SKE).sum(dim='lev')/rho_regrid.sum(dim='lev')
    SKE.name='SKE'
    SKE.attrs={'long_name':'SKE','units':'J Kg^-1'}

    SKEDOT=((Eddy_Tend_Zon*ushear + Eddy_Tend_Mer*vshear)*dPbyg).sum(dim='lev')
    SKEDOT.name='SKEDOT'
    SKEDOT.attrs={'long_name':'dp/g Integral(-d/dp([uw]-[u][w])*u_shear - d/dp([vw]-[v][w])*v_shear)','units':'W m-2'}
    
    skedot_dataset=xr.merge([SKE,SKEDOT,upwp,vpwp,uw,vw,u_baro,v_baro,Eddy_Flux_Zon,Eddy_Flux_Mer,ushear,Eddy_Tend_Zon,Eddy_Tend_Mer,vshear])

    #return SKEDOT
    return skedot_dataset

def SKEDot_from_4deg(time_selection,lon_selection,lat_selection):
    '''Main difference between this and SKEDot function is this reads 4deg hourly 
     averaged data from weather.rsmas.so no regridding performed here.'''
    if lon_selection.start <0:
       lon_start=360.0+lon_selection.start
       if lon_selection.stop <0:
           lon_stop=360.0+lon_selection.stop
       lon_selection=slice(lon_start,lon_stop)
       
    da_4deg=load_4deg_dataset()
    if isinstance(time_selection,slice):
        u_regrid=da_4deg.U.sel(time=time_selection).sel(lon=lon_selection,lat=lat_selection)
        v_regrid=da_4deg.V.sel(time=time_selection).sel(lon=lon_selection,lat=lat_selection)
        w_regrid=da_4deg.W.sel(time=time_selection).sel(lon=lon_selection,lat=lat_selection)
    
        uw_regrid=da_4deg.WU.sel(time=time_selection).sel(lon=lon_selection,lat=lat_selection)
        vw_regrid=da_4deg.WV.sel(time=time_selection).sel(lon=lon_selection,lat=lat_selection)
    else:
        u_regrid=da_4deg.U.sel(time=time_selection,method='nearest').sel(lon=lon_selection,lat=lat_selection)
        v_regrid=da_4deg.V.sel(time=time_selection,method='nearest').sel(lon=lon_selection,lat=lat_selection)
        w_regrid=da_4deg.W.sel(time=time_selection,method='nearest').sel(lon=lon_selection,lat=lat_selection)
    
        uw_regrid=da_4deg.WU.sel(time=time_selection,method='nearest').sel(lon=lon_selection,lat=lat_selection)
        vw_regrid=da_4deg.WV.sel(time=time_selection,method='nearest').sel(lon=lon_selection,lat=lat_selection)
        
    T_sample=xr.open_dataset('http://weather.rsmas.miami.edu/repository/opendap/synth:1142722f-a386-4c17-a4f6-0f685cd19ae3:L0c1TlIvVF9yOTB4NDVfMXRpbWUubmM0/entry.das')
    T_sample=T_sample.isel(time=0).sel(lon=lon_selection,lat=lat_selection)['T']
    T_sample=T_sample.drop('time')

    P=T_sample.lev*100.0
    rho_regrid=(1/(T_sample*287.06))*P 

    
    upwp=uw_regrid-u_regrid*w_regrid
    upwp.name='upwp'
    vpwp=vw_regrid-v_regrid*w_regrid
    vpwp.name='vpwp'
    
    Eddy_Flux_Zon=upwp*rho_regrid
    Eddy_Flux_Mer=vpwp*rho_regrid
    Eddy_Flux_Zon.name='Eddy_Flux_Zon'
    Eddy_Flux_Mer.name='Eddy_Flux_Mer'
#make it a dataset for easy function application on all variables 
    Eddy_Flux=xr.merge([Eddy_Flux_Zon,Eddy_Flux_Mer])
   
    dp=u_regrid.lev
    dp=dp*100.0
    dp=np.gradient(dp)
    dPbyg=dp/9.8
    dPbyg=xr.DataArray(dPbyg,coords={'lev':u_regrid.lev},dims='lev')
    axisint=1 if len(np.shape(Eddy_Flux_Zon))>3 else 0
    Eddy_Flux_Tend=Eddy_Flux.apply(np.gradient,axis=axisint)
    Eddy_Flux_Tend=Eddy_Flux_Tend/dPbyg
   
    Eddy_Tend_Zon=Eddy_Flux_Tend['Eddy_Flux_Zon']
    Eddy_Tend_Zon.name='Eddy_Tend_Zon'

    Eddy_Tend_Mer=Eddy_Flux_Tend['Eddy_Flux_Mer']
    Eddy_Tend_Mer.name='Eddy_Tend_Mer'
 
     
    u_baro=(rho_regrid*u_regrid).sum(dim='lev')/rho_regrid.sum(dim='lev')
    u_baro.name='ubaro'
    v_baro=(rho_regrid*v_regrid).sum(dim='lev')/rho_regrid.sum(dim='lev')
    v_baro.name='vbaro'
    

    ushear=u_regrid-u_baro
    ushear.name='ushear'
    vshear=v_regrid-v_baro
    vshear.name='vshear'

    SKE=0.5*(ushear*ushear+vshear*vshear)
    SKE=(rho_regrid*SKE).sum(dim='lev')/rho_regrid.sum(dim='lev')
    SKE.name='SKE'
    SKE.attrs={'long_name':'SKE','units':'J Kg^-1'}

    SKEDOT=((Eddy_Tend_Zon*ushear + Eddy_Tend_Mer*vshear)*dPbyg).sum(dim='lev')
    SKEDOT.name='SKEDOT'
    SKEDOT.attrs={'long_name':'dp/g Integral(-d/dp([uw]-[u][w])*u_shear - d/dp([vw]-[v][w])*v_shear)','units':'W m-2'}
    
    skedot_dataset=xr.merge([SKEDOT,SKE,u_baro,v_baro,Eddy_Flux_Zon,Eddy_Flux_Mer,ushear,Eddy_Tend_Zon,Eddy_Tend_Mer,vshear])

    #return SKEDOT
    return skedot_dataset
import requests
from PIL import Image,ImageDraw
import itertools
from bisect import bisect_left
import io
requests.urllib3.disable_warnings()
def G5NR_image(variable,yyyymmddhhmm,lon=0,lat=0,dlon=180,dlat=90,save_global=False,scale_image=1.0,geoviews=False):
    def frange(start, end, num_of_elements):
        delta=float(end-start)/(num_of_elements-1)
        newend=end+delta
        retl=start
        while retl < newend :
            yield retl
            retl += delta
    def getUrl(yyyymmddhhmm,variable):    
        baseurl="https://g5nr.nccs.nasa.gov/static/naturerun/fimages"
        d="/"

        stringList=[baseurl,variable.upper(),"Y"+yyyymmddhhmm[0:4],"M"+yyyymmddhhmm[4:6],"D"+yyyymmddhhmm[6:8]]
        stringList+=[variable.lower()+"_globe_c1440_NR_BETA9-SNAP_"+yyyymmddhhmm[0:8]+"_"+yyyymmddhhmm[8:]+"z.png"]
        url=d.join(stringList)
        return url
        
    def download_file(url):
        local_filename = url.split('/')[-1]

        r = requests.get(url, stream=True,verify=False)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk:
                    f.write(chunk)
        return local_filename
    
    def download_bytes(url):
        local_filename = url.split('/')[-1]

        r = requests.get(url, stream=True,verify=False)
        byt = io.BytesIO()
        for chunk in r.iter_content(chunk_size=512): 
            if chunk: # filter out keep-alive new chunks
                 byt.write(chunk)
        return byt

    url=getUrl(yyyymmddhhmm,variable)


    try:
        if save_global:
            f=download_file(url) #saves and returns filename
        else:
            f = download_bytes(url) #on the fly bytes
        oimg = Image.open(f)
        size = oimg.size
    except IOError:
        return url+' Not available '
    dlon=min(dlon,180.0)
    dlat=min(dlat,90.0)
    
    if scale_image<1.0:
        latbylon=size[1]/size[0]
        nlon=int(size[0]*scale_image)
        oimg=oimg.resize((nlon,int(nlon*latbylon)))
        size = oimg.size
    
    if dlon==180.0 and dlat==90.0:
        latbylon=size[1]/size[0]
        nlon=int(size[0]*scale_image)
        return oimg.resize((nlon,int(nlon*latbylon)))
    lats=list(frange(-90,90,size[1]))
    lons=list(frange(-17.5,342.5,size[0]))
    
    new_im = Image.new('RGB', (size[0],size[1]))
    x_offset = 0
    xind=bisect_left(lons,180)
    
    images=[oimg.crop((xind,0,size[0],size[1])),oimg.crop((0,0,xind-1,size[1]))]
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    oimg.close()
    oimg=new_im
        
    lons=list(frange(-180.0,180.0,size[0]))
        
    if lon>180.0: 
        lon=lon-360.0

    lon_st=bisect_left(lons,max(lon-dlon,-180.0)) 
    lon_ct=bisect_left(lons,min(max(lon,-180.0),180.0))
    lon_en=bisect_left(lons,min(lon+dlon,180.0))
    
    lat_en=size[1]-(bisect_left(lats,max(lat-dlat,-90.0))+1)
    lat_ct=size[1]-(bisect_left(lats,lat)+1)
    lat_st=size[1]-(bisect_left(lats,min(lat+dlat,90.0))+1)

    
    lonrange=(lons[lon_st],lons[lon_en])
    latrange=(lats[bisect_left(lats,max(lat-dlat,-90.0))],lats[bisect_left(lats,min(lat+dlat,90.0))])

    lonboxst=bisect_left(lons,min(max(lon-2,-180.0),180.0))
    lonboxen=bisect_left(lons,min(max(lon+2,-180.0),180.0))
    latboxen=size[1]-(bisect_left(lats,max(min(90,lat+2),-90)+1))
    latboxst=size[1]-(bisect_left(lats,max(min(90,lat-2),-90)+1))
  
    draw = ImageDraw.Draw(oimg)
    box=(lonboxst, latboxen, lonboxen, latboxst)
    width=int(4*scale_image)
    for _ in range(width):
        draw.rectangle(box,outline='Red')
        box=(box[0]+1,box[1]+1,box[2]+1,box[3]+1)
        
    cpd_img=oimg.crop((lon_st,lat_st,lon_en,lat_en))
    oimg.close()
    size=cpd_img.size
    #if scale_image<1.0:
    #    latbylon=size[1]/size[0]
    #    nlon=int(size[0]*scale_image)
    #    cpd_img=cpd_img.resize((nlon,int(nlon*latbylon)))
    
    if geoviews:
        #in future use geoviews.RGB, right now buggy so hv.RGB
        label=variable.upper()
        group='Lon: '+format(lon,"0.1f")+' Lat: '+format(lat,"0.1f")+' Time: '+str(yyyymmddhhmm)
        img=hv.RGB(np.array(cpd_img),label=label,group=group)#.redim.range(Longitude=lonrange,Latitude=latrange)
        return img(plot={'xaxis':None,'yaxis':None,'width':600})
    else:
        return cpd_img  
