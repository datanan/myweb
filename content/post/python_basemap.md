---
author: "xiaoxie"
date: 2019-05-16
title: "python basemap lcc projection"
tags: [
    "mask",
    "basemap",
    "lcc"
]
categories: [
    "python",
]
---


# 【写在前面】

兰伯特投影的不同设置方法，width,height的方法关于中心经纬度对称，推荐使用设置llcrnrlat的方法
maskout.py用于裁剪特定的区域，注意pcolor画图方式设置vcplot=False，contourf等设置为True


## 这是代码

```
# -- coding=utf-8 --
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
#                                   AnnotationBbox)
import regionmask
import geopandas
import pandas as pd
import maskout
import math
import warnings
warnings.filterwarnings("ignore")

def latlon_grid(bmap, lon_int, lat_int, labels='lb', **kwargs):
    '''Draws a lat-lon grid in an easy way.

    Some default values are taken from rcParams instead of 'black' (color) and
    1.0 (linewidth) which is the default in Basemap.

    In Basemap, the label pad is computed in projection units. Now you can use
    the keyword argument 'labelpad' to control this separation in points. If
    not specified then this value is taken from rcParams.

    Arguments:

    bmap -- Basemap object.
    lon_int, lat_int -- Difference in degrees from one longitude or latitude to
                        the next.
    labels -- String specifying which margins will be used to write the labels.
              If None, no label will be shown.
              It is assummed that left/right margins (i.e. Y axes) correspond
              to latitudes and top/bottom (X axes) to longitudes. It is valid
              every combination of the characters 't' | 'b' | 'l' | 'r'
              (top|bottom|left|right).
              Ex: 'lrb' means that the longitude values will appear in bottom
              margin and latitudes in left and right.
    **kwargs -- Other arguments to drawparallels, drawmeridians and plt.text.
                labelpad has units of points.
    '''
    # Proccesses arguments and rcParams for defult values
    if 'color' not in kwargs:
        kwargs['color'] = plt.rcParams['grid.color']
    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = plt.rcParams['grid.linewidth']
    if 'labelpad' in kwargs:
        padx = pady = kwargs['labelpad']
        del kwargs['labelpad']
    else:
        pady = plt.rcParams['xtick.major.pad']
        padx = plt.rcParams['ytick.major.pad']
    if 'size' in kwargs:
        xfontsize = yfontsize = kwargs['size']
        del kwargs['size']
    elif 'fontsize' in kwargs:
        xfontsize = yfontsize = kwargs['fontsize']
        del kwargs['fontsize']
    else:
        xfontsize = plt.rcParams['xtick.labelsize']
        yfontsize = plt.rcParams['ytick.labelsize']
    # Vectors of coordinates
    lon0 = bmap.lonmin // lon_int * lon_int
    lat0 = bmap.latmin // lat_int * lat_int
    lon1 = bmap.lonmax // lon_int * lon_int
    lat1 = bmap.latmax // lat_int * lat_int
    nlons = (lon1 - lon0) / lon_int + 1
    nlats = (lat1 - lat0) / lat_int + 1
    assert nlons / int(nlons) == 1, nlons
    assert nlats / int(nlats) == 1, nlats
    lons = np.linspace(lon0, lon1, int(nlons))
    lats = np.linspace(lat0, lat1, int(nlats))
    # If not specified then computes de label offset by 'labelpad'
    xos = yos = None
    if 'xoffset' in kwargs:
        xos = kwargs['xoffset']
    if 'yoffset' in kwargs:
        yos = kwargs['yoffset']
    if xos is None and yos is None:
        # Page size in inches and axes limits
        fig_w, fig_h = plt.gcf().get_size_inches()
        points = plt.gca().get_position().get_points()
        x1, y1 = tuple(points[0])
        x2, y2 = tuple(points[1])
        # Width and height of axes in points
        w = (x2 - x1) * fig_w * 72
        h = (y2 - y1) * fig_h * 72
        # If the aspect relation is fixed then compute the real values
        if bmap.fix_aspect:
            aspect = bmap.aspect * w / h
            if aspect > 1:
                w = h / bmap.aspect
            elif aspect < 1:
                h = w * bmap.aspect
        # Offset in projection units (meters or degrees)
        xos = padx * (bmap.urcrnrx - bmap.llcrnrx) / w
        yos = pady * (bmap.urcrnry - bmap.llcrnry) / h
    # Set the labels
    latlabels = [False] * 4
    lonlabels = [False] * 4
    if labels is not None:
        pst = {'l': 0, 'r': 1, 't': 2, 'b': 3}
        lst = {'l': latlabels, 'r': latlabels, 't': lonlabels, 'b': lonlabels}
        for i in labels.lower():
            lst[i][pst[i]] = True
    # Draws the grid
    bmap.drawparallels(lats, labels=latlabels, fontsize=yfontsize,
                       xoffset=xos, yoffset=yos, **kwargs)
    bmap.drawmeridians(lons, labels=lonlabels, fontsize=xfontsize,
                       xoffset=xos, yoffset=yos, **kwargs)

def plot_rec(bmap, lower_left, upper_left, lower_right, upper_right):
    xs = [lower_left[0], upper_left[0],
          upper_right[0],lower_right[0],lower_left[0]]
    ys = [lower_left[1], upper_left[1],
          upper_right[1],lower_right[1],lower_left[1]]
    bmap.plot(xs, ys, color="r",latlon = True)

def fill_maxwell(ax):
        Nlignes=300
        Ncol=300
        img = np.zeros((Nlignes,Ncol,4))
        dx = 2.0/(Ncol-1)
        dy = 1.0/(Nlignes-1)
        for i in range(Ncol-1):
            for j in range(Nlignes-1):
                x = -1.0+i*dx
                y = j*dy
                v = y
                r = (x+1-v)/2.0
                b = 1.0-v-r
                if (r>=0) and (r<=1.0) and (v>=0) and (v<=1.0) and (b>=0) and (b<=1.0):
                    r,v,b = 1/max(r,v,b)*r,1/max(r,v,b)*v,1/max(r,v,b)*b
                    img[j][i] = np.array([r,v,b,1.0])
                else:
                    img[j][i] = np.array([1.0,1.0,1.0,0.0])
        a = 1.0/math.sqrt(3)
        fig = ax.imshow(img,origin='lower',extent=[-a,a,0.0,1.0])
        # ax.plot([0,a/2.,-a/2.,0],[0,0.5,0.5,0],'w--',linewidth=2,)
        # ax.annotate("", xy=( a/2.,0.5), xytext=( -a ,0.),  arrowprops=dict(arrowstyle="->"))
        # ax.annotate("", xy=(-a/2.,0.5), xytext=( a,0.),   arrowprops=dict(arrowstyle="->"))
        # ax.annotate("", xy=( 0,1.),     xytext=( 0,0.),  color='white', arrowprops=dict(arrowstyle="->"))
        # ax.annotate("", xy=( a ,0.),    xytext=( -a/2.,0.5) ,color='white',arrowprops=dict(arrowstyle="->"))
        # ax.annotate("", xy=(-a,0.),     xytext=( a/2.,0.5) ,color='white',arrowprops=dict(arrowstyle="->"))
        # ax.annotate("", xy=( 0,0.),     xytext=( 0,1.),     arrowprops=dict(arrowstyle="->"))        
        # #         
        # ax.plot([0,a/2.],[1/3.,0.5],'w--',linewidth=2,)
        # ax.plot([0,-a/2.],[1/3.,0.5],'w--',linewidth=2,)
        # ax.plot([0,0.],[1/3.,0.],'w--',linewidth=2,)  
        fsize = 7      
        ax.text(-0.78, -0.15, "$\Delta{TEMP}$", fontsize=fsize, color='w') #,rotation=30
        ax.text( 0.38, -0.15, "$\Delta{RADF}$", fontsize=fsize, color='w')  ## ,rotation=-30
        ax.text( -0.15,  1.05, "$\Delta{VPD}$", fontsize=fsize, color='w')

        # ax.text(  a/2.+0.05,0.6, "TEMP\n-100%"   , fontsize=fsize, color='w',rotation=30)
        # ax.text( -a/2-0.3,0.6, "RADF\n-100%"   , fontsize=fsize, color='w',rotation=-30)
        # ax.text( -0.1,  -0.2, " VPD\n-100%", fontsize=fsize, color='w')
                        
        ax.axis('off')
        return fig

def piefig(ax,fracs):
    labels = '$\Delta{RADF}$', '$\Delta{TEMP}$', '$\Delta{VPD}$'
    # A standard pie plot
    wedges, texts, autotexts =ax.pie(fracs, labels=labels, autopct='%.0f%%',textprops=dict(color="w",fontsize=7, weight="bold")\
        , colors=['r','b','g'],shadow=True)
    plt.setp(autotexts, size=5, weight="bold")
    plt.setp(texts, size=7, weight="bold")
    # ax.set_title("Dominant factor",fontsize=5)
    return ax


def regfig(lat,lon,data,fracs,z_min=0.,z_max=1.5,figname="regcmfig",title=None,units=None):
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.85,0.85]) 
    #m = Basemap(projection='lcc', \
    #            area_thresh=1000.,rsphere=(6378137.00,6356752.3142),\
    #            width=5640000,height=4140000,\
    #            lat_0=36.,lon_0=107.,lat_1=25.,lat_2=50., ax=ax)
    m = Basemap(
       llcrnrlat=lats[0,0],urcrnrlat=lats[-1,-1],llcrnrlon=lons[0,0],urcrnrlon=lons[-1,-1],
       resolution='i', area_thresh=1000.,
       lat_0=36.,lon_0=107.,lat_1=25.,lat_2=50.,
       projection='lcc')

    # m.fillcontinents(color='gray',lake_color='aqua')
    m.drawmapboundary(fill_color='dimgray')  #dimgray
    xi, yi = m(lon, lat)
    cs = m.imshow(data,interpolation='nearest')
    clip = maskout.shp2clip(cs,ax,m,'sixregion',"China",vcplot = True)
    # cs = m.pcolor(xi, yi, np.squeeze(data), cmap="rainbow",vmin=z_min, vmax=z_max)
    # cs = m.pcolor(xi, yi, np.squeeze(data),norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
    #                                           vmin=z_min, vmax=z_max), cmap="coolwarm")
    # bounds = np.linspace(z_min, z_max, 20)
    # norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # cs = m.pcolor(xi, yi, np.squeeze(data), cmap="coolwarm",norm=norm)  ### Pcolor with a log scale
    latlon_grid(m, 10, 5, labels='lb', labelpad=10, linewidth=0.) #dashes=[1, 3]
    # parallels = np.arange(0.,60,5.)
    # m.drawparallels(parallels, color='k', linewidth=.01,
    #  zorder=0, dashes=[1, 1], labels=[1,0,0,0], fmt='%g') # , fontsize=10
    # meridians = np.arange(80.,180.,10.)
    # m.drawmeridians(meridians, color='k', linewidth=1.0,
    #  zorder=0, dashes=[1, 1], labels=[0,0,0,1], fmt='%g') # , fontsize=10
    #  
    llcrnrlon =  103.
    urcrnrlon =  122.
    llcrnrlat =  21.
    urcrnrlat =  40.
    lower_left = (llcrnrlon, llcrnrlat)
    lower_right= (urcrnrlon, llcrnrlat)
    upper_left = (llcrnrlon, urcrnrlat)
    upper_right= (urcrnrlon, urcrnrlat)
    # plot_rec(m, lower_left, upper_left, lower_right, upper_right)
    m.readshapefile('coastline_withoutchina','whatevername',color='black')
    # m.readshapefile('cnmap/cnmap','whatevername',color='black')
    # m.readshapefile('country1','whatevername',color='black')
    m.readshapefile('cnmap/chinaprov','whatevername',color='black')
    
    # cbar_num_format = "%0.1f"
    # cbar = m.colorbar(cs, pad="5%", format=cbar_num_format)
    # cbar = m.colorbar(cs, location='bottom', pad="10%", format=cbar_num_format)
    # if(units): cbar.set_label(units)
    # if(title): plt.title(title)
    if(title): fig.text(0.1, 0.95, title,fontsize="medium")
    # if(title): fig.text(0.8, 0.25, title,fontsize="large")
    # plt.show()
    left, bottom, width, height = 0.78, 0.19, 0.15, 0.15
    ax2 = fig.add_axes([left, bottom, width, height])
    fig2 = fill_maxwell(ax2)
    left, bottom, width, height = 0.17, 0.18, 0.15, 0.15
    ax3 = fig.add_axes([left, bottom, width, height])
    fig3 = piefig(ax3,fracs)    
    # plt.savefig(figname, dpi=300)
    with PdfPages(figname+'.pdf') as pdf:
        pdf.savefig(fig,bbox_inches='tight',pad_inches = 0.05 )    

meteo_file = "contrb.nc"
fh = Dataset(meteo_file)

# 获取每个变量的值
lons = fh.variables['xlon'][4:73,7:95]
lats = fh.variables['xlat'][4:73,7:95]
# mparsp = fh.variables['rad'][4:73,7:95]
# mtassp = fh.variables['temp'][4:73,7:95]
# msmsp  = fh.variables['vpd'][4:73,7:95]
# 
gpp   = fh.variables['gpp'][::,4:73,7:95]
parsp = fh.variables['gpar'][::,4:73,7:95]
tassp = fh.variables['gtas'][::,4:73,7:95]
smsp  = fh.variables['gsm'][::,4:73,7:95]
ter   = fh.variables['ter'][::,4:73,7:95]
parau = fh.variables['tpar'][::,4:73,7:95]
tasau = fh.variables['ttas'][::,4:73,7:95]
smau  = fh.variables['tsm'][::,4:73,7:95]
nee   = fh.variables['nee'][::,4:73,7:95]
par   = fh.variables['npar'][::,4:73,7:95]
tas   = fh.variables['ntas'][::,4:73,7:95]
smq   = fh.variables['nsm'][::,4:73,7:95]
fh.close()

mgpp   = np.nanmean(gpp   ,axis=0)
mparsp = np.nanmean(parsp,axis=0)
mtassp = np.nanmean(tassp,axis=0)
msmsp  = np.nanmean(smsp ,axis=0)
mter   = np.nanmean(ter   ,axis=0)
mparau = np.nanmean(parau,axis=0)
mtasau = np.nanmean(tasau,axis=0)
msmau  = np.nanmean(smau ,axis=0)
mnee   = np.nanmean(nee   ,axis=0)
mpar   = np.nanmean(par  ,axis=0)
mtas   = np.nanmean(tas  ,axis=0)
msmq   = np.nanmean(smq  ,axis=0)

conv = 86400000.
mgpp   *= conv
mparsp *= conv
mtassp *= conv
msmsp  *= conv
mter   *= conv
mparau *= conv
mtasau *= conv
msmau  *= conv
mnee   *= conv
mpar   *= conv
mtas   *= conv
msmq   *= conv

### mask data using regionmask package and shapely
### first create mask array
df = geopandas.read_file('/home/tjwang/regcm/modis/ndvi/sixregion.shp')
numbers = [0, 1, 2,3,4,5]
names = list(df.Name)
abbrevs = list(df.ID)
CHmask = regionmask.Regions_cls('CHmask', numbers, names, abbrevs, df.geometry)
mask = CHmask.mask(lons, lats, xarray=False)
### mask data: 
### note: currently there are six subregions in our shpfile, 
# gpp = np.ma.masked_where(np.isnan(mask), gpp) ## only china area
# gpp = np.ma.masked_where(mask!=1, gpp)  ### only region 1 -- Central
# region index:
# 0      North
# 1    Central
# 2  SouthEast
# 3  NorthEast
# 4  SouthWest
# 5  NorthWest 
# def regionmean(mask,var):
#     res,sgm = [],[]
#     for i in range(6):
#         varmask = np.ma.masked_where(mask!=i, var)
#         var1d = varmask.flatten()
#         res.append(np.nanmean(var1d))
#         sgm.append(np.nanvar(var1d))
#     return res,sgm

# avg_mgpp  , var_mgpp   = regionmean(mask, mgpp  )
# avg_mparsp, var_mparsp = regionmean(mask, mparsp)
# avg_mtassp, var_mtassp = regionmean(mask, mtassp)
# avg_msmsp , var_msmsp  = regionmean(mask, msmsp )
# avg_mter  , var_mter   = regionmean(mask, mter  )
# avg_mparau, var_mparau = regionmean(mask, mparau)
# avg_mtasau, var_mtasau = regionmean(mask, mtasau)
# avg_msmau , var_msmau  = regionmean(mask, msmau )
# avg_mnee  , var_mnee   = regionmean(mask, mnee  )
# avg_mpar  , var_mpar   = regionmean(mask, mpar  )
# avg_mtas  , var_mtas   = regionmean(mask, mtas  )
# avg_msmq  , var_msmq   = regionmean(mask, msmq  )
# regions = ("North","Central","SouthEast","NorthEast","SouthWest","NorthWest")

# x = np.c_[avg_gpp,avg_par,avg_tas,avg_sm]

# avgx = np.c_[avg_mgpp  ,avg_mparsp,avg_mtassp,avg_msmsp ,avg_mter  ,avg_mparau,avg_mtasau,avg_msmau ,avg_mnee  ,avg_mpar  ,avg_mtas  ,avg_msmq]
# avgx = [avg_mgpp  , var_mgpp  ,avg_mparsp, var_mparsp,avg_mtassp, var_mtassp,avg_msmsp , var_msmsp ,avg_mter  , var_mter  ,avg_mparau, var_mparau,avg_mtasau, var_mtasau,avg_msmau , var_msmau ,avg_mnee  , var_mnee  ,avg_mpar  , var_mpar  ,avg_mtas  , var_mtas  ,avg_msmq  , var_msmq]
# # varx = np.c_[var_mgpp  ,var_mparsp,var_mtassp,var_msmsp ,var_mter  ,var_mparau,var_mtasau,var_msmau ,var_mnee  ,var_mpar  ,var_mtas  ,var_msmq]
# pd.DataFrame(avgx).to_csv('avg_contrib.csv',index=None)
# pd.DataFrame(varx).to_csv('var_contrib.csv',index=None)
# fig, ax = plt.subplots()
# y_pos = np.arange(len(regions))
# ax.barh(y_pos, avg_gpp, xerr=var_gpp, align='center',
#         color='green', ecolor='black')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(regions)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('GPP Difference')
# ax.set_title('a')
# plt.savefig('region1',dpi=300)
# print(avg_gpp)
# print(avg_par,avg_tas,avg_sm)
# N = 6
# ind = np.arange(N)    # the x locations for the groups
# width = 0.4       # the width of the bars: can also be len(x) sequence
# p1 = plt.bar(ind, avg_par, width, yerr=var_par,color='orangered')
# p2 = plt.bar(ind, avg_tas, width,
#           bottom=avg_par, yerr=var_tas,color='orange')
# p3 = plt.bar(ind, avg_sm, width,
#           bottom=np.sum([avg_par,avg_tas], axis = 0), yerr=var_sm,color='deepskyblue')
# plt.ylabel('GPP Difference')
# plt.title('a')
# plt.xticks(ind, regions)
# plt.xlim(-.4,6.4)
# # plt.yticks(np.arange(0, 81, 10))
# plt.legend((p1[0], p2[0],p3[0]), ('RAD', 'TEMP', 'SM'))
# plt.savefig('region2',dpi=300)

# x = np.c_[avg_gpp,avg_par,avg_tas,avg_sm]
# print(x.shape)
# n_bins = 6
# fig, ax0 = plt.subplots()
# colors = ['red', 'green', 'tan', 'lime']
# labels = ['GPP','RAD', 'TEMP', 'SM']
# ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=labels)
# ax0.legend(prop={'size': 10})
# ax0.set_title('bars with legend')
# plg.savefig('region2',dpi=300)
# 
# data1=round((data1-min(min(data1)))*255/(max(max(data1))-min(min(data1))));%归一化
# mparspmax = max(mparsp.min(), mparsp.max(), key=abs)
# mtasspmax = max(mtassp.min(), mtassp.max(), key=abs)
# msmspmax  = max(msmsp.min(), msmsp.max(), key=abs)

### GPP
# mparspmax = abs(max(mparsp.min(), mparsp.max(), key=abs))
# mtasspmax = abs(max(mtassp.min(), mtassp.max(), key=abs))
# msmspmax  = abs(max(msmsp.min(), msmsp.max(), key=abs))
# print(mparsp.min(),mparsp.max())
# print(mparspmax,mtasspmax,msmspmax)
# mparsp = (mparsp+mparspmax)/mparspmax/2.
# mtassp = (mtassp+mtasspmax)/mtasspmax/2.
# msmsp  = (msmsp+msmspmax)/msmspmax/2.
mparsp = abs(mparsp)#/mparspmax
mtassp = abs(mtassp)#/mtasspmax
msmsp  = abs(msmsp) # /msmspmax
mparsp = np.where(abs(mgpp)<.01, 1., mparsp)
mtassp = np.where(abs(mgpp)<.01, 1., mtassp)
msmsp  = np.where(abs(mgpp)<.01, 1., msmsp )
### TER
# mparspmax = abs(max(mparau.min(), mparau.max(), key=abs))
# mtasspmax = abs(max(mtasau.min(), mtasau.max(), key=abs))
# msmspmax  = abs(max(msmau.min(), msmau.max(), key=abs))
# print(mparau.min(),mparau.max())
# print(mparspmax,mtasspmax,msmspmax)
mparau = abs(mparau)#/mparspmax
mtasau = abs(mtasau)#/mtasspmax
msmsp  = abs(msmau) # /msmspmax
mparau = np.where(abs(mter)<.01, 1., mparau)
mtasau = np.where(abs(mter)<.01, 1., mtasau)
msmau  = np.where(abs(mter)<.01, 1., msmau )

print(np.nanmax(mparsp))
print(np.nanmin(mparsp))
# mparsp = (mparsp + 1.)/2.
# mtassp = (mtassp + 1.)/2.
# msmsp  = (msmsp  + 1.)/2.
#### NEE
# mparspmax = abs(max(mpar.min(), mpar.max(), key=abs))
# mtasspmax = abs(max(mtas.min(), mtas.max(), key=abs))
# msmspmax  = abs(max(msmq.min(), msmq.max(), key=abs))
# mparsp = (mpar+mparspmax)/mparspmax/2.
# mtassp = (mtas+mtasspmax)/mtasspmax/2.
# msmsp  = (msmq+msmspmax)/msmspmax/2.
mpar = abs(mpar)#/mparspmax
mtas = abs(mtas)#/mtasspmax
msmq  = abs(msmq) # /msmspmax
mpar = np.where(abs(mnee)<.01, 1., mpar)
mtas = np.where(abs(mnee)<.01, 1., mtas)
msmq = np.where(abs(mnee)<.01, 1., msmq )

chparsp = mparsp[:,:,np.newaxis]
chtassp = mtassp[:,:,np.newaxis]
chsmsp  = msmsp[:,:,np.newaxis]
chgpp = np.c_[chparsp,chtassp,chsmsp] 
sts = np.argmax(abs(chgpp), axis=2)
sts = np.ma.masked_where(np.isnan(mask), sts) ## only china area
sts = np.ma.masked_where(abs(mgpp)<.01, sts)
a = np.sum( sts == 0 )
b = np.sum( sts == 1 )
c = np.sum( sts == 2 )
print(a,b,c)
fracsgpp = [a,b,c]

mparsp = np.where(abs(mgpp)<.01, 1., mparsp)
mtassp = np.where(abs(mgpp)<.01, 1., mtassp)
msmsp  = np.where(abs(mgpp)<.01, 1., msmsp )

mparsp = mparsp[:,:,np.newaxis]
mtassp = mtassp[:,:,np.newaxis]
msmsp  = msmsp[:,:,np.newaxis]
print(mparsp.shape)
cgpp = np.c_[mparsp,msmsp,mtassp]   ### par: red; vpd: green; temp: blue
cgpp  = cgpp * (1/np.max(cgpp,axis=2))[:,:,np.newaxis]
regfig(lats,lons,cgpp, fracsgpp,z_min=-1.,z_max=1.,figname="contribgpp",units=None,title="a) Main factors for GPP")

chparsp = mparau[:,:,np.newaxis]
chtassp = mtasau[:,:,np.newaxis]
chsmsp  = msmau[:,:,np.newaxis]
chgpp = np.c_[chparsp,chtassp,chsmsp] 
sts = np.argmax(abs(chgpp), axis=2)
sts = np.ma.masked_where(np.isnan(mask), sts) ## only china area
sts = np.ma.masked_where(abs(mter)<.01, sts)
a = np.sum( sts == 0 )
b = np.sum( sts == 1 )
c = np.sum( sts == 2 )
print(a,b,c)
fracsgpp = [a,b,c]
mparau = mparau[:,:,np.newaxis]
mtasau = mtasau[:,:,np.newaxis]
msmau  = msmau[:,:,np.newaxis]
cgpp = np.c_[mparau,msmau,mtasau]   ### par: red; vpd: green; temp: blue
cgpp  = cgpp * (1/np.max(cgpp,axis=2))[:,:,np.newaxis]
regfig(lats,lons,cgpp, fracsgpp,z_min=-1.,z_max=1.,figname="contribter",units='GPP Difference',title="b) Main factors for TER")

chparsp = mpar[:,:,np.newaxis]
chtassp = mtas[:,:,np.newaxis]
chsmsp  = msmq[:,:,np.newaxis]
chgpp = np.c_[chparsp,chtassp,chsmsp] 
sts = np.argmax(abs(chgpp), axis=2)
sts = np.ma.masked_where(np.isnan(mask), sts) ## only china area
sts = np.ma.masked_where(abs(mnee)<.01, sts)
a = np.sum( sts == 0 )
b = np.sum( sts == 1 )
c = np.sum( sts == 2 )
print(a,b,c)
fracsgpp = [a,b,c]
mpar = mpar[:,:,np.newaxis]
mtas = mtas[:,:,np.newaxis]
msmq  = msmq[:,:,np.newaxis]
cgpp = np.c_[mpar,msmq,mtas]   ### par: red; vpd: green; temp: blue
cgpp  = cgpp * (1/np.max(cgpp,axis=2))[:,:,np.newaxis]
regfig(lats,lons,cgpp, fracsgpp,z_min=-1.,z_max=1.,figname="contribnee",units='GPP Difference',title="c) Main factors for NEE")

```


# mask特定的区域 [maskout.py]
```python
#coding=utf-8
###################################################################################################################################
#####This module enables you to maskout the unneccessary data outside the interest region on a matplotlib-plotted output instance
####################in an effecient way,You can use this script for free     ########################################################
#####################################################################################################################################
#####USAGE: INPUT  include           'originfig':the matplotlib instance##
#                                    'ax': the Axes instance
#                                    'm': the Basemap instance
#                                    'shapefile': the shape file used for generating a basemap A
#                                    'region':the name of a region of on the basemap A,outside the region the data is to be maskout
#                   optional         'clabel': clabel instance
#                                    'vcplot': vector instance flag
#           OUTPUT    is             'clip' :the the masked-out or clipped matplotlib instance.
import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from collections import Iterable
def shp2clip(originfig,ax,m,shpfile,region, clabel = None, vcplot = None):
    sf = shapefile.Reader(shpfile)
    vertices = []
    codes = []
    for shape_rec in sf.shapeRecords():
        ####这里需要找到和region匹配的唯一标识符，record[]中必有一项是对应的。
        #if shape_rec.record[3] == region:   #####在country1.shp上，对中国以外的其他国家或地区进行maskout
        #if shape_rec.record[7] in region:  #####在bou2_4p.shp上，对中国的某几个省份或地区之外的部分进行maskout
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i+1]):
                vertices.append(m(pts[j][0], pts[j][1]))
            codes += [Path.MOVETO]
            codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
            codes += [Path.CLOSEPOLY]
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)



#    if isinstance(originfig,Iterable):
#        for ivec in originfig:
#            ivec.set_clip_path(clip)
#    else:
#        originfig.set_clip_path(clip)



    if vcplot:
        if isinstance(originfig,Iterable):
            for ivec in originfig:
                ivec.set_clip_path(clip)
        else:
            originfig.set_clip_path(clip)
    else:
        for contour in originfig.collections:
            contour.set_clip_path(clip)

    if  clabel:
        clip_map_shapely = ShapelyPolygon(vertices)
        for text_object in clabel:
            if not clip_map_shapely.contains(ShapelyPoint(text_object.get_position())):
                text_object.set_visible(False)    

    return clip
```