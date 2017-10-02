import xarray as xr
import numpy as np
def genlon_bins(nlon):
    """mainly to match cdo calculations"""
    """watch out when close to 180"""
    import itertools
    start=-180.0
    stop=180.0
    dy=(stop-start)/float(nlon)
    ra=itertools.count(start-dy/2,dy)
    return [ra.next() for i in range(nlon+1)]
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
    return [ra.next() for i in range(nlat+1)]
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
    #print("#############")
    #print(regrided)
    #regrided=regrided.rename({'lat_bins':'lat','lon_bins':'lon'})
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
 
def upwp_u_w_uw(u,w,uw,nlat,nlon):
    u_reg=regrid(u,nlat,nlon)
    w_reg=regrid(w,nlat,nlon)
    uw_reg=regrid(uw,nlat,nlon)

    upwp_reg=uw_reg-(u_reg*w_reg)
    #upwp_reg=upwp_reg.rename({'U':'UpWp'})
    upwp_reg.name='UpWp'
    
    return xr.merge([u_reg,w_reg,uw_reg,upwp_reg])
  
def upwp_u_w(u,w,nlat,nlon):
    up=subgrid(u,nlat,nlon)
    wp=subgrid(w,nlat,nlon)
    upwp=up*wp
    upwp.name='UpWp'
    upwp_reg=regrid(upwp,nlat,nlon)

    u_reg=regrid(u,nlat,nlon)
    w_reg=regrid(w,nlat,nlon)
    uw_reg=regrid(u*w,nlat,nlon)
    uw_reg.name='wu'
    return xr.merge([u_reg,w_reg,uw_reg,upwp_reg]) 

def SKEDot(rho,u,v,w,nlon,nlat):
    u_regrid=regrid(u,nlon,nlat)
    v_regrid=regrid(v,nlon,nlat)
    w_regrid=regrid(w,nlon,nlat) 
    rho_regrid=regrid(rho,nlon,nlat) 

    up=subgrid(u,nlon,nlat)
    wp=subgrid(w,nlon,nlat)
    vp=subgrid(u,nlon,nlat)
   
    rhoupwp=rho*up*wp
    rhovpwp=rho*up*wp
    
           
    Eddy_Flux_Zon=regrid(rhoupwp,nlon,nlat)
    Eddy_Flux_Mer=regrid(rhoupwp,nlon,nlat)
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

    u_baro=(rho_regrid*u_regrid).sum(dim='lev')/rho_regrid.sum(dim='lev')
    v_baro=(rho_regrid*v_regrid).sum(dim='lev')/rho_regrid.sum(dim='lev')

    u_baro=(rho_regrid*u_regrid).sum(dim='lev')/rho_regrid.sum(dim='lev')
    v_baro=(rho_regrid*v_regrid).sum(dim='lev')/rho_regrid.sum(dim='lev')

    ushear=u_regrid-u_baro
    vshear=v_regrid-v_baro

    SKE=ushear*ushear+vshear*vshear
    SKE=(rho_regrid*SKE).sum(dim='lev')/rho_regrid.sum(dim='lev')
    SKE.name='SKE'
    SKE.attrs={'long_name':'SKE','units':'J Kg^-1'}

    SKEDOT=(Eddy_Flux['Eddy_Flux_Zon']*ushear/dPbyg + Eddy_Flux['Eddy_Flux_Mer']*vshear/dPbyg).sum(dim='lev')
    SKEDOT.name='SKEDOT'
    SKEDOT.attrs={'long_name':'dp/g Integral(-d/dp([uw]-[u][w])*u_shear - d/dp([vw]-[v][w])*v_shear)','units':'W m-2'}
    return SKEDOT

