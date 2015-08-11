# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:54:43 2014
This routine can plot both the observed and modeled drifter tracks.
It has various options including how to specify start positions, how long to track, 
whether to generate animation output, etc. See Readme.
@author: Bingwei Ling
Derived from previous particle tracking work by Manning, Muse, Cui, Warren.
"""

import sys
import pytz
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from track_functions import get_drifter,get_fvcom,get_roms,draw_basemap,uniquecolors,clickmap, points_between, points_square,extend_square,totdis
from matplotlib import animation

st_run_time = datetime.now() # Caculate execution time with en_run_time
############################### Options #######################################
'''
Option 1: Drifter track.
Option 2: Specify the start point.
Option 3: Specify the start point with simulated map.
Option 4: Area(box) track.          
'''
######## Hard codes ##########
Option = 1 # 1,2,3,4
print 'Option %d'%Option
MODEL = 'GOM3'     # 'ROMS', 'GOM3','massbay','30yr'
GRIDS = ['GOM3','massbay','30yr']    # All belong to FVCOM. '30yr' works from 1977/12/31 22:58 to 2014/1/1 0:0
depth = 1    # depth below ocean surface, positive
track_days = 1     #MODEL track time(days)
track_way = 'forward'    # Three options: backward, forward and both. 'both' only apply to Option 2 and 3.
image_style = 'plot'      # Two option: 'plot', animation
# You can track form now by specify start_time = datetime.now(pytz.UTC) 
start_time = datetime(2015,8,10,12,45,0,0,pytz.UTC)#datetime.now(pytz.UTC) 
end_time = start_time + timedelta(track_days)
model_boundary_switch = 'OFF' # OFF or ON. Only apply to FVCOM
streamline = 'OFF'
save_dir = '/home/bling/Documents/Results/'

################################## Option ####################################
if Option==1:
    drifter_ID = '157410703'#152300811 
    # if raw data, use "drift_X.dat";if want to get drifter data in database, use "None"
    INPUT_DATA = 'drift_X.dat'#'drift_jml_2015_1.dat'      

if Option==2: # user specified pts
    point1 = (38.734,-73.685)  # 42.1, -70.6 Point data structure:(lat,lon)
    extend_style = 'line' #or 'square'
    if extend_style=='line':
        point2 = ()#41.686903, -70.665452#
        (st_lat,st_lon)=points_between(point1,point2,0) # The number parameter is the number between the two points.
    if extend_style=='square':  #return a squre points(lats,lons) on center point
        side_length = 0.1
        (st_lat,st_lon) = points_square(point1,side_length)

if Option == 3: # click on a map , press ENTER ended.
    numpts=9 # Points  added on the map at most
    [st_lon,st_lat]=clickmap(numpts) # gets lat/lon by clicking map

if Option == 4:
    addpointway = 'square'  #Two options: random, square
    if addpointway=='random':
        num = 33
        st_lat = np.random.uniform(41.9,42.1,num)[:]
        st_lon = np.random.uniform(-70.4,-70.6,num)[:]
    if addpointway=='square':
        centerpoint = (42.0,-70.5); radius = 0.1; number = 3
        (st_lat,st_lon) = extend_square(centerpoint,radius,number)

############################## Common codes ###################################
loop_length = []
fig = plt.figure() #figsize=(16,9)
ax = fig.add_subplot(111)
points = {'lats':[],'lons':[]}  # collect all points we've gained

############################## Option 1 ###################################
if Option == 1:
    #depth = -1
    drifter_points = dict(lon=[],lat=[])
    model_points = dict(lon=[],lat=[])
    print "Drifter: %s %s %d days"%(drifter_ID,track_way,track_days)
    start_time = datetime.now(pytz.UTC)-timedelta(track_days) #datetime(2015,1,24,0,0,0,0,pytz.UTC)
    
    drifter = get_drifter(drifter_ID, INPUT_DATA)
    dr_points = drifter.get_track(start_time,track_days)
    drifter_points['lon'].extend(dr_points['lon']); drifter_points['lat'].extend(dr_points['lat'])
    print "drifter points: ",len(dr_points['lon']),'\nlast point(',dr_points['lat'][-1],',',dr_points['lon'][-1],')'
    #np.savez('drifter_points.npz',lon=drifter_points['lon'],lat=drifter_points['lat'])
    start_time = dr_points['time'][-1]
    end_time = dr_points['time'][-1] + timedelta(track_days)
    if track_way=='backward':
        end_time = start_time 
        start_time = end_time - timedelta(track_days)  #''' 
    if MODEL in GRIDS:
        get_obj =  get_fvcom(MODEL)
        print dr_points['time'][-1]
        url_fvcom = get_obj.get_url(start_time,end_time)
        b_points = get_obj.get_data(url_fvcom) # b_points is model boundary points.
        point,num = get_obj.get_track(dr_points['lon'][-1],dr_points['lat'][-1],depth,track_way)
        
    if MODEL=='ROMS':        
        end_time = dr_points['time'][-1] + timedelta(track_days)
        get_obj = get_roms()
        url_roms = get_obj.get_url(dr_points['time'][-1],end_time)
        get_obj.get_data(url_roms)
        point = get_obj.get_track(dr_points['lon'][-1],dr_points['lat'][-1],depth,track_way)#,DEPTH
        if len(point['lon'])==1:
            print 'Start point on the land or out of Model area.'
            sys.exit('Invalid point')
    model_points['lon'].extend(point['lon']); model_points['lat'].extend(point['lat'])
    loop_length.append(len(model_points['lon']))
    #np.savez('model_points.npz',lon=model_points['lon'],lat=model_points['lat'])
    dripath = totdis(dr_points['lon'],dr_points['lat'])
    modpath = totdis(point['lon'],point['lat'])
    discrepancy = modpath-dripath
    print 'Model path length: ',modpath,'\nDrifter path length: ',dripath,'\nDiscrepancy: ',discrepancy,discrepancy/dripath*100
   
    points['lats'].extend(drifter_points['lat']); points['lons'].extend(drifter_points['lon'])
    points['lats'].extend(model_points['lat']); points['lons'].extend(model_points['lon'])  
            
    ############################ 1 Features Option #########################    
    if model_boundary_switch=='ON':
        po = b_points.T
        ax.plot(po[0],po[1],'bo',markersize=3)
    
    if streamline == 'ON':
        lonpps,latpps,US,VS,speeds = get_obj.streamlinedata(points,depth,track_way)
        np.savez('streamline.npz',lonpps=lonpps,latpps=latpps,US=US,VS=VS,speeds=speeds)
        image_style = 'animation'
        
    ########################### Plot #####################################
    plt.title('Drifter: {0} {1}'.format(drifter_ID,MODEL))    
    #colors=uniquecolors(len(points['lats'])) #config colors
    an2 = str(dr_points['time'][-1].strftime('%m/%d-%H:%M'))    
    
    if image_style=='plot':#facecolor=colors[i]'''
        draw_basemap(ax, points)  # points is using here
        ax.annotate(an2,xy=(dr_points['lon'][-1],dr_points['lat'][-1]),xytext=(dr_points['lon'][-1]+0.01*track_days,
                    dr_points['lat'][-1]+0.01*track_days),fontsize=6,arrowprops=dict(arrowstyle="fancy")) 
        ax.plot(drifter_points['lon'],drifter_points['lat'],'bo-',markersize=6,label='Drifter')
        ax.plot(model_points['lon'],model_points['lat'],'ro',markersize=4,label='Model')
        plt.legend(loc=4)
    
    if image_style=='animation':
        if streamline == 'ON':
            def animate(n): #del ax.collections[:]; del ax.lines[:]; ax.cla();ax.clf()
                ax.cla()
                ax.plot(drifter_points['lon'],drifter_points['lat'],'bo-',markersize=6,label='Drifter')
                if streamline == 'ON':
                    plt.streamplot(lonpps[n],latpps[n],US[n],VS[n], color=speeds[n],arrowsize=4,cmap=plt.cm.cool,density=2.0)              
                if n==0:#facecolor=colors[i]'''
                    ax.annotate(an2,xy=(dr_points['lon'][-1],dr_points['lat'][-1]),xytext=(dr_points['lon'][-1]+0.01*track_days,
                                dr_points['lat'][-1]+0.01*track_days),fontsize=6,arrowprops=dict(arrowstyle="fancy"))
                ax.plot(model_points['lon'][:n+1],model_points['lat'][:n+1],'ro-',markersize=6,label=MODEL)
                draw_basemap(ax, points)  # points is using here
            anim = animation.FuncAnimation(fig, animate, frames=max(loop_length), interval=1000)#        
            plt.clim(vmin=0, vmax=1)
            plt.colorbar()
            
        else:
            draw_basemap(ax, points)  # points is using here
            ax.plot(drifter_points['lon'],drifter_points['lat'],'bo-',markersize=6,label='Drifter')
            ax.annotate(an2,xy=(dr_points['lon'][-1],dr_points['lat'][-1]),xytext=(dr_points['lon'][-1]+0.01*track_days,
                        dr_points['lat'][-1]+0.01*track_days),fontsize=6,arrowprops=dict(arrowstyle="fancy"))
            def animate(n): # the function of the animation
                ax.plot(model_points['lon'][:n+1],model_points['lat'][:n+1],'ro-',markersize=6,label=MODEL)
            anim = animation.FuncAnimation(fig, animate, frames=max(loop_length), interval=250) #

#####################Option 2|3 ########################
if Option==2 or Option==3:
    stp_num = len(st_lat)
    lon_set = [[]]*stp_num; lat_set = [[]]*stp_num
    colors=uniquecolors(stp_num) #config colors
    print 'You added %d points.' % stp_num,st_lon,st_lat
    if track_way=='backward':
        end_time = start_time 
        start_time = end_time - timedelta(track_days)  #'''    
    
    if MODEL in GRIDS:
        get_obj = get_fvcom(MODEL)
        url_fvcom = get_obj.get_url(start_time,end_time)
        b_points = get_obj.get_data(url_fvcom) 
        
        '''if model_boundary_switch=='ON': # b_points is model boundary points.
            lonb = lonc[b_index]; latb = latc[b_index]        
            b_points = np.vstack((lonb.flatten(),latb.flatten())).T#'''
        for i in range(stp_num):
            print 'Running the %dth of %d drifters.'%(i+1,stp_num)
            point,nu = get_obj.get_track(st_lon[i],st_lat[i],depth,track_way)
            lon_set[i] = point['lon']; lat_set[i] = point['lat']
            loop_length.append(len(point['lon']))
        
        if track_way == 'both':
            track_way = 'backward'
            image_style = 'plot'
            end_time = start_time
            start_time = end_time - timedelta(track_days)            
            url_fvcom1 = get_obj.get_url(start_time,end_time)
            for k in range(stp_num):
                print 'Running the %dth of %d drifters.'%(k+1,stp_num)
                point,nu = get_obj.get_track(st_lon[k],st_lat[k],depth,track_way)
                lon_set[k].append(point['lon']); lat_set[k].append(point['lat'])
                loop_length.append(len(point['lon']))
            track_way = 'both'
    
    if MODEL=='ROMS': 
        get_obj = get_roms()
        url_roms = get_obj.get_url(start_time,end_time)
        get_obj.get_data(url_roms)
        for i in range(stp_num):
            point = get_obj.get_track(st_lon[i],st_lat[i],depth,track_way)
            if len(point['lon'])==1:
                print 'Start point on the land or out of Model area.'
                sys.exit('Invalid point')
            lon_set[i] = point['lon']; lat_set[i] = point['lat']
            loop_length.append(len(point['lon']))
            
    if not track_way=='both': 
        np.savez('model_points.npz',lon=lon_set,lat=lat_set) 
        
    if track_way == 'both': #The last element is a list of backward points.        
        for i in range(stp_num):
            bl = len(lon_set[i])-1
            points['lons'].extend(lon_set[i][:bl])
            points['lats'].extend(lat_set[i][:bl])
            points['lons'].extend(lon_set[i][bl])
            points['lats'].extend(lat_set[i][bl])        
    else: 
        for i in range(stp_num):
            points['lons'].extend(lon_set[i])
            points['lats'].extend(lat_set[i])
    print 'Points quantity: ',len(points['lons'])
    
    ######################### 2|3 Features Option #############################    
    if streamline == 'ON':
        lonpps,latpps,US,VS,speeds = get_obj.streamlinedata(points,depth,track_way)
        np.savez('streamline.npz',lonpps=lonpps,latpps=latpps,US=US,VS=VS,speeds=speeds)
        image_style = 'animation' 
        
    if model_boundary_switch=='ON':
        po = b_points.T
        ax.plot(po[0],po[1],'bo',markersize=3)
        
    ######################### Plot #############################    
    plt.suptitle('%s to %s forecast\n-1m depth'%(start_time.strftime('%D'),end_time.strftime('%D')))
    
    if image_style=='plot':
        draw_basemap(ax, points)
        if track_way == 'both':
            ax.plot(lon_set[0][0],lat_set[0][0],'ro-',markersize=3,label='forward')
            ax.plot(lon_set[0][0],lat_set[0][0],'go-',markersize=3,label='backward')
            for j in range(stp_num):
                bl = len(lon_set[j])-1
                ax.annotate('Start %d'%(j+1), xy=(lon_set[j][0],lat_set[j][0]),xytext=(lon_set[j][0]+0.01*stp_num,
                            lat_set[j][0]+0.01*stp_num),fontsize=6,arrowprops=dict(arrowstyle="fancy")) #facecolor=colors[i]'''
                ax.plot(lon_set[j][:bl],lat_set[j][:bl],'go-',markersize=4)#markerfacecolor='r',
                ax.plot(lon_set[j][bl],lat_set[j][bl],'ro-',markersize=4)
            plt.legend(loc=4)
        else: 
            for j in range(stp_num):
                ax.annotate('Start %d'%(j+1), xy=(lon_set[j][0],lat_set[j][0]),xytext=(lon_set[j][0]+0.05,
                            lat_set[j][0]+0.03),fontsize=6,arrowprops=dict(arrowstyle="fancy")) #facecolor=colors[i]'''
                ax.plot(lon_set[j],lat_set[j],'o-',color=colors[j],markersize=3,label='Start %d'%(j+1)) #markerfacecolor='r',
    
    if image_style=='animation':
        if streamline == 'ON':
            def animate(n): #del ax.collections[:]; del ax.lines[:]; ax.cla();ax.clf()
                ax.cla()
                if streamline == 'ON':
                    plt.streamplot(lonpps[n],latpps[n],US[n],VS[n], color=speeds[n],arrowsize=4,cmap=plt.cm.cool,density=2.0)
                for j in range(stp_num):
                    if n==0:#facecolor=colors[i]'''
                        ax.annotate('Start %d'%(j+1), xy=(lon_set[j][0],lat_set[j][0]),xytext=(lon_set[j][0]+0.01*stp_num,
                                    lat_set[j][0]+0.01*stp_num),fontsize=6,arrowprops=dict(arrowstyle="fancy")) 
                    if n<len(lon_set[j]): #markerfacecolor='r',
                        ax.plot(lon_set[j][:n+1],lat_set[j][:n+1],'o-',color=colors[j],markersize=4,label='Start %d'%(j+1))
                draw_basemap(ax, points)  # points is using here
            anim = animation.FuncAnimation(fig, animate, frames=max(loop_length), interval=1000)#        
            plt.clim(vmin=0, vmax=1)
            plt.colorbar()
        else:
            draw_basemap(ax, points)
            for j in range(stp_num):
                def animate(n): #del ax.collections[:]; del ax.lines[:]; ax.cla();ax.clf()
                    if n==0:#facecolor=colors[i]'''
                        ax.annotate('Start %d'%(j+1), xy=(lon_set[j][0],lat_set[j][0]),xytext=(lon_set[j][0]+0.01*stp_num,
                                    lat_set[j][0]+0.01*stp_num),fontsize=6,arrowprops=dict(arrowstyle="fancy")) 
                    if n<len(lon_set[j]): #markerfacecolor='r',
                        ax.plot(lon_set[j][:n+1],lat_set[j][:n+1],'o-',color=colors[j],markersize=4,label='Start %d'%(j+1))
                anim = animation.FuncAnimation(fig, animate, frames=max(loop_length), interval=250)
                               
#####################Option 4 ########################
if Option==4:
    image_style = 'animation'
    hitland = 0; onland = 0
    stp_num = len(st_lat)
    lon_set = [[]]*stp_num; lat_set = [[]]*stp_num;
    print 'You added %d points.' % stp_num,st_lon,st_lat
    #start_time = datetime.now(pytz.UTC)+timedelta(track_time)  #datetime(2015,2,10,12,0,0,0,pytz.UTC)#
    #end_time = start_time + timedelta(track_days)
    if track_way=='backward':
        end_time = start_time 
        start_time = end_time - timedelta(track_days)  #'''
     
    if MODEL in GRIDS:            
        get_obj = get_fvcom(MODEL)
        url_fvcom = get_obj.get_url(start_time,end_time)
        b_points = get_obj.get_data(url_fvcom)# b_points is model boundary points.        
        
        for i in range(stp_num):
            print 'Running the %dth of %d drifters.'%(i+1,stp_num)
            point,nu = get_obj.get_track(st_lon[i],st_lat[i],depth,track_way)
            #point,nu = get_obj.get_track(st_lon[i],st_lat[i],lonc,latc,u,v,b_points,track_way)
            lon_set[i] = point['lon']; lat_set[i] = point['lat']
            loop_length.append(len(point['lon']))
            if nu==0:
                onland+=1
            if nu==1:
                hitland+=1             
        p = float(hitland)/float(stp_num-onland)*100
        print "%d points, %d hits the land, ashore percent is %.f%%."%(stp_num-onland,hitland,int(round(p)))
        np.savez('model_points.npz',lon=lon_set,lat=lat_set)
    for i in range(stp_num):
        points['lons'].extend(lon_set[i])
        points['lats'].extend(lat_set[i])
    
    ######################### 4 Features Option #############################
    if model_boundary_switch=='ON':
        po = b_points.T
        ax.plot(po[0],po[1],'bo',markersize=3)
    if streamline == 'ON':
        lonpps,latpps,US,VS,speeds = get_obj.streamlinedata(points,depth,track_way)
        np.savez('streamline.npz',lonpps=lonpps,latpps=latpps,US=US,VS=VS,speeds=speeds)
    
    ######################### Plot #######################################       
    
    colors=uniquecolors(stp_num) #config colors
    plt.suptitle('%.f%% simulated drifters ashore\n%d days, %d m, %s'%
              (int(round(p)),track_days,depth,start_time.strftime("%d-%b-%Y")))
    if streamline == 'ON':
        def animate(n): #del ax.collections[:]; del ax.lines[:]; ;
            ax.cla()
            draw_basemap(ax, points)  # points is using here
            for j in xrange(stp_num):
                ax.plot(lon_set[j][0],lat_set[j][0],color=colors[j],marker='x',markersize=4)
                if n>=len(lon_set[j]):
                    ax.plot(lon_set[j][-1],lat_set[j][-1],'o',color=colors[j],markersize=5)
                if n<5:                
                    if n<len(lon_set[j]):
                        ax.plot(lon_set[j][:n+1],lat_set[j][:n+1],'o-',color=colors[j],markersize=4)#,label='Depth=10m'            
                if n>=5:
                    if n<len(lon_set[j]):
                        ax.plot(lon_set[j][n-4:n+1],lat_set[j][n-4:n+1],'o-',color=colors[j],markersize=4)
        anim = animation.FuncAnimation(fig, animate, frames=max(loop_length), interval=1000) #        
    
    else:
        draw_basemap(ax, points)  # points is using here
        def animate(n): #del ax.collections[:]; del ax.lines[:]; ax.cla(); ax.lines.remove(line)        
            del ax.lines[:]        
            for j in xrange(stp_num):
                ax.plot(lon_set[j][0],lat_set[j][0],color=colors[j],marker='x',markersize=4)
                if n>=len(lon_set[j]):
                    ax.plot(lon_set[j][-1],lat_set[j][-1],'o',color=colors[j],markersize=5)
                if n<5:                
                    if n<len(lon_set[j]):
                        ax.plot(lon_set[j][:n+1],lat_set[j][:n+1],'o-',color=colors[j],markersize=4)#,label='Depth=10m'            
                if n>=5:
                    if n<len(lon_set[j]):
                        ax.plot(lon_set[j][n-4:n+1],lat_set[j][n-4:n+1],'o-',color=colors[j],markersize=4)
        anim = animation.FuncAnimation(fig, animate, frames=max(loop_length),interval=250) #, 
    
##################################### The End ##########################################
en_run_time = datetime.now()
print 'Take '+str(en_run_time-st_run_time)+' running the code. End at '+str(en_run_time)
### Save #########
#plt.legend(loc=4)
if image_style=='plot':
    plt.savefig(save_dir+'%s-%s_%s'%(MODEL,track_way,en_run_time.strftime("%d-%b-%Y_%H:%M")),
                dpi=400,bbox_inches='tight')
if image_style=='animation':#ffmpeg,imagemagick,mencoder fps=20'''
    anim.save(save_dir+'%s-%s_%s.gif'%(MODEL,track_way,en_run_time.strftime("%d-%b-%Y_%H:%M")),
              writer='imagemagick',dpi=250) #,,,fps=1
plt.show()
