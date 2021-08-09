import numpy as np
import itertools as it
#from numba import jit

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
from matplotlib.patches import Ellipse,Polygon,FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

import networkx as nx
import os
import shutil

from shapely import geometry
from shapely.geometry.polygon import LinearRing

# next steps:
# check what noise_phi is doing and how to make it understandable (so that noise=0 means no noise)
# add option for a hole in the middle of milling state


def getKey0(item):
    return item[0]

def getKey1(item):
    return item[1]

def getKey5(item):
    return item[5]

def get_ellipse_line_intersection_points(eyes,tps,w):
    ''' given two points of the line (eyes and tp) calculates
        the points at which this line intersects with an ellipse
        of length 1 and width w with center at the origin and orientation
        along the positive x-axis, returns points as 2x2 array, 
        index1: x/y, 
        index2: which intersection point,
        if only 1 intersections found both entries are equal,
        if no intersections are found, entries are np.nan'''
    x1=eyes[0]
    y1=eyes[1]
    x2=tps[0]
    y2=tps[1]
    a=0.5
    b=w/2.
    dd=((x2-x1)**2/(a**2)+(y2-y1)**2/(b**2))
    ee=(2.*x1*(x2-x1)/(a**2)+2.*y1*(y2-y1)/(b**2))
    ff=(x1**2/(a**2)+y1**2/(b**2)-1.)
    determinant=ee**2-4.*dd*ff
    float_epsilon=0.00001
    zeromask=abs(determinant)>=1000.*float_epsilon
    determinant*=zeromask
    t=(np.array([(-ee-np.sqrt(determinant))/(2.*dd),(-ee+np.sqrt(determinant))/(2.*dd)]))
    mask=np.array(t>0.,dtype=float)
    mask[mask==0.]=np.nan
    x=mask*(x1+(x2-x1)*t)
    y=mask*(y1+(y2-y1)*t)
    return np.array([x,y])

#@jit(cache=True,parallel=True,nopython=True)    
def remove_self_intersections(inters,n):
    ''' used to remove intersections of ray emitted from ellipse i's eye and intersecting with 
        ellipse i's boundary when detecting all intersections of those rays with all other ellipses,
        inters is array of interception points with indices ijklm
        i: x/y [2], 
        j: which intersection [2], 
        k: on which ellipse [n], 
        l: for which viewer [n], 
        m: for which ray [2(n-1)]'''

    for i in range(n):
        inters[:,:,i,i,:]=np.nan
    return inters

def get_closest_id(r,out,n):

    ''' used to find the closest intersection point on a ray emitted from and ellipses eye,
        r is numpy array with indices jklm as follows:
        j: which intersection [2], 
        k: on which ellipse [n], 
        l: for which viewer [n], 
        m: for which ray [2(n-1)]'''

    for j,k in it.product(range(n),range((n-1)*2)):
            if np.isnan(r[:,:,j,k]).all():
                out[j,k]=np.nan
            else:
                out[j,k]=np.nanargmin(r[:,:,j,k],axis=1)[1]
    return out


def psi_go(w,r,theta,phi,mainaxis=0.5):

    '''calculates where the tangent points lie on the ellipse, return the corresponding angles,
    these can be translated in to coordinates via using the function ellipsepoints_forgo()
    (as found in simon leblanc's go code, IMPORTANT: l is the length of the main axis,
    not the position of the eye'''
    l=mainaxis
    w=w/2.0
    aa=np.sqrt(-2.0*l*l*w*w + (l*l + w*w)*r*r + (w*w - l*l)*r*r*np.cos(2.0*(theta - phi)))/np.sqrt(2.0)
    bb= w*r*np.cos(theta - phi) - l*w
    psi1=2.0*np.arctan2(aa-l*r*np.sin(theta - phi),bb)
    psi2= -2.0*np.arctan2(aa+l*r*np.sin(theta - phi),bb)
    return [psi1,psi2]

def numberofrings(nn):

        lower_estimate=nn/np.pi
        upper_estimate=(np.sqrt(4.*np.pi*nn+1)+1)/2.*np.pi
        return int(np.floor(lower_estimate)), int(np.floor(upper_estimate))

def cast_to_pm_pi(a):

        b = (a+np.pi)%(2.*np.pi)
        b -= np.pi
        return b

def ellipse_intersect(pos1,phi1,pos2,phi2,w):

    ellipses = [(pos1[0], pos1[1], 1, w/2.0, phi1), (pos2[0], pos2[1], 1, w/2.0, phi2)]
    a, b = ellipse_polyline(ellipses)
    inter = intersections(a, b)
    return inter

def ellipse_polyline(ellipses, n=100):

    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a/2.0 * ca * ct - b * sa * st
        p[:, 1] = y0 + a/2.0 * sa * ct + b * ca * st
        result.append(p)
    return result

def point_inside(p,s):
    pnt=geometry.Point(p)
    shape=geometry.Polygon(s)
    return pnt.within(shape)

def intersections(a, b):

	ea = LinearRing(a)	
	eb = LinearRing(b)
	mp = ea.intersects(eb)
	return mp

def is_closer(pt,linepoint1,linepoint2):
    r1=linepoint1[0]
    phi1=linepoint1[1]
    r2=linepoint2[0]
    phi2=linepoint2[1]
    phi=pt[1]
    m=(r2*np.sin(phi2)-r1*np.sin(phi1))/(r2*np.cos(phi2)-r1*np.cos(phi1))
    r=(r1*np.sin(phi1)-m*r1*np.cos(phi1))/(np.sin(phi)-m*np.cos(phi))
    if r<=pt[0]:
        ret=False
    else:
        ret=True
    return ret

def merge_segments(visible, hidden):
    # used to decide which of two segments occludes the other
    # the naming should be exactly opposite (hidden is actually blocking visible)
    
    if visible[0]!=visible[1]:
        ret=[visible]
        a=smallestSignedAngleBetween(visible[0],hidden[0])
        b=smallestSignedAngleBetween(visible[1],hidden[0])
        c=smallestSignedAngleBetween(visible[0],hidden[1])
        d=smallestSignedAngleBetween(visible[1],hidden[1])
        e=smallestSignedAngleBetween(visible[0],visible[1])
        f=smallestSignedAngleBetween(hidden[0],hidden[1])   
        if (a>=0 and a<=e) or (c>=0 and c<=f):
            if a>=0:
                if d>=0:
                    if b<=0:
                        ret=[[visible[0],hidden[0]]]
                else:
                    ret=[[visible[0],hidden[0]],[hidden[1],visible[1]]]
            else:
                if d>=0:
                    ret=[] 
                else:
                    if c>=0:
                        ret=[[hidden[1],visible[1]]]
    else:
        ret=[]
    return ret


def line_intersect(x1a,y1a,x1b,y1b,x2a,y2a,x2b,y2b):
    # finds the point where two lines intersect.
    # lines are given by two points each:
    # line 1 goes through (x1a,y1a) and (x1b, y1b)
    # the same goes for line2
    
	point=None
	if (x1b-x1a)!=0:
		if (x2b-x2a)!=0:
			m1=(y1b-y1a)/(x1b-x1a)
			m2=(y2b-y2a)/(x2b-x2a)
			if m1!=m2:
				b1=y1a-m1*x1a
				b2=y2a-m2*x2a
				x=(b2-b1)/(m1-m2)
				y=m1*x+b1
				point=[x,y]
			else:
				print('lines are parallel')
		else:
			x=x2b
			m1=(y1b-y1a)/(x1b-x1a)
			b1=y1a-m1*x1a
			y=m1*x+b1
			point=[x,y]			
	else:
		if (x2b-x2a)!=0:
			x=x1b
			m2=(y2b-y2a)/(x2b-x2a)
			b2=y2a-m2*x2a
			y=m2*x+b2
			point=[x,y]
		else:
			print('lines are parallel')
	if point!=None:
		point=np.array(point)
	return point
	

def smallestSignedAngleBetween(x,y): 
    #returns smallest of the two angles from x to y 
    tau=2*np.pi
    a = (x - y) % tau
    b = (y - x) % tau
    return -a if a < b else b

def ellipsepoints_forgo(r,theta,phi,psi,w,l=0.5):
    #calculates cartesian coordinates for a point on an ellipse 
    # with long axis 1, short axis w, ellipse center at r,theta
    # that is given by the ellipse parameter psi
    
    x=r*np.cos(theta) + l*np.cos(phi)*np.cos(psi) + w*l*np.sin(phi)*np.sin(psi)
    y=r*np.sin(theta) + l*np.sin(phi)*np.cos(psi) - w*l*np.cos(phi)*np.sin(psi)
    return [x,y]

def get_partnerid(arrayentry,searcharray):
    test=np.where(np.array(searcharray[:,2],dtype=int)==int(arrayentry[2]))[0]
    for k in test:
        if arrayentry[3]-searcharray[k,3]:
            partner_id=int(k)
    return partner_id

def subjpol_to_objcart(r,theta,pos,phi):
    # takes in a point, r, theta from the polar coordinates 
    # with center at pos and orientation phi
    # returns a point in cartesian coordinates (same 
    # coordinate system that pos is given in)
	rot_mat_back=np.array([[np.cos(-phi),np.sin(-phi)],[-np.sin(-phi),np.cos(-phi)]])
	pt_subj=[r*np.cos(theta),r*np.sin(theta)]
	pt_obj=np.dot(rot_mat_back,pt_subj)+pos
	return pt_obj

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



class Swarm:

	'''#####################################################################
	This class stores all data on the swarm on ellipses
		pos:            position of the eye of each ellipse (2xN array)
		pos_center:     position of the center of each ellipse (2xN array)
		phi:            orientation of each ellipse (axis 1 and pos. x-axis)
		w:              width of the ellipse, float
		l:              twice the offset of the eye on the ellipse main axis
						from the center, float (l=-1 back, l=1 front, l=0 center)
		n:              number of ellipses (just for convenience), int
		md:             array of shape n*n, euclidean distance of ellipses 
						centers to another
		intersections:  list of indices of ellipses that intersect, format 
						is [[e1,e2,e3],[e4,e5,e6]] if pairs of intersecting 
						ellipses are (e1,e4), (e2,e5), (e3,e6)
		
		tp_subj_pol:    array of shape [2,n,n] containing the 2 tangent 
						points (as 2d arrays) for all combinations of 
						ellipses. Indices [i,j,k]:
							i: 1st or second tangent point
							j: id of target (tps lie on this ellipse)
							k: id of observer (this ellipses contains the eye)
						entries are 2d arrays containing r, theta of tangent
						point in polar coordinates with the eye of the ob-
						server in the origin and theta=0 on pos x-axis
						
		tp_obj_cart:    array of shape [2,n,n] containing the 2 tangent 
						points (as 2d arrays) for all combinations of 
						ellipses. Indices [i,j,k]:
							i: 1st or second tangent point
							j: target (tps lie on this ellipse)
							k: observer (this ellipses contains the eye)
						entries are 2d arrays containing x,y of tangent
						point in the objective cartesian coordinates (same
						as used for pos)
		segments:       dictionary with ellipse index pairs as keys (i,j), 
						containing a list of upper and lower boundaries of
						segments of i that are visible to j
		
		unclassified:   array (n*n) used to save if a target ellipse i has been
						assigned a final angular area it obtains in the vi-
						sual field of observer j yet. If it has, 
						unclassified[i,j]=False
		vis_angles:	array (n*n) used to save the visual angle an individual i would have in the 
						visual field of individual j (second index) if not obscured
						by any others
		
	#####################################################################'''


	def __init__(self,N=40,setup='grid',pos=None,pos_center=None,pos_offset=0.,phi=None,w=0.1,l=0.,bl=1.,dist=2.,noise_pos=0.1,noise_phi=0.9):
		''' When an instance of swarm is initialized this sets:
			- pos: the position of the eye,
			- pos_center: position of the geometric center of each ellipse,
			- phi: orientations
			- w: width of the ellipses
			- l: half of the distance of the eye from the center along the
				 main axis of the ellipse of length 1

		if positions and orientations are given, these are used to create the 
		swarm and any input in N is ignored.
		If no pos and orient are given, they are generated for N ellipses according the input
		parameters. 
		'''
		if pos_center is not None and pos is None:
			if np.shape(pos_center)[0]!=2:
				if np.shape(pos_center)[1]==2:
					pos_center=pos_center.T
					self.pos_center=pos_center
				else:
					print('positions need to be of shape [2,N] or ([N,2]')
			else:
				self.pos_center=pos_center
			if phi is not None:
	#			#print(len(phi),np.shape(pos_center)[1])
				if len(phi)==np.shape(pos_center)[1]:
					self.pos_center-=np.array([pos_offset/2.0*np.cos(phi),pos_offset/2.0*np.sin(phi)])
					self.pos=self.pos_center-np.array([-l/2.0*np.cos(phi),-l/2.0*np.sin(phi)])
					self.phi=phi	
				else:
					print('Length of orientations array must correspond to number of given positions')
			else:
				print('Please set orientations')
		elif pos is not None and pos_center is None:
			if np.shape(pos)[0]!=2:
				if np.shape(pos)[1]==2:
					pos=pos.T
					self.pos=pos
				else:
					print('positions need to be of shape [2,N] or ([N,2]')
			else:
				self.pos=pos
			if phi is not None:
				#print(len(phi),np.shape(pos)[1])
				if len(phi)==np.shape(pos)[1]:
					self.pos-=np.array([pos_offset/2.0*np.cos(phi),pos_offset/2.0*np.sin(phi)])
					self.pos_center=self.pos+np.array([-l/2.0*np.cos(phi),-l/2.0*np.sin(phi)])
					self.phi=phi	
				else:
					print('Length of orientations array must correspond to number of given positions')
			else:
				print('Please set orientations')
		elif pos is not None and pos_center is not None:
			print('Please use only position of center or of eye to create swarm.')
		else:
			#initiate positions and orientations according ta N and setup
			pos,phi=self.gen_init_config(setup,N,noise_pos,dist,noise_phi,w)
			self.pos_center=pos
			self.pos=pos-np.array([-l/2.0*np.cos(phi),-l/2.0*np.sin(phi)])
			self.phi=phi	
		
		self.w=w
		self.l=l
		self.bl=bl
		self.reset_calculated_variables()
	
	def gen_init_config(self,state,nn,noise_int,d,kappa,w):
		#print('N=%i'%nn)
		if state=='grid':
			#print('generating grid configuration')
			n=int(np.floor(np.sqrt(nn)))
			xlen=n
			ylen=n
			number=n*n
			grid_x=np.linspace(d,d*xlen,xlen,endpoint=True)
			grid_y=np.linspace(d,d*ylen,ylen,endpoint=True)
			x,y=np.meshgrid(grid_x,grid_y)
			pos=np.array([x.flatten(),y.flatten()])
			if n<np.sqrt(nn):
				for i in range(nn-number):
					extra=np.array([d*(xlen+1+np.floor(i/n)),d*(i%n+1)]).reshape(2,1)
					pos=np.hstack([pos,extra])
			orientations=np.random.vonmises(0.0,kappa,nn)
			noise=(np.random.random((2,nn))-np.ones((2,nn))*0.5)*2.0*noise_int*d
			pos=pos+noise
			return pos,orientations
		elif state=='dense shifted grid':
			d_w=d*w
			n=int(np.floor(np.sqrt(nn)))
			xlen=n
			ylen=n
			number=n*n
			grid_x=np.linspace(d,d*xlen,xlen,endpoint=True)
			grid_y=np.linspace(d_w,d_w*ylen,ylen,endpoint=True)
			x,y=np.meshgrid(grid_x,grid_y)
			x[0:-1:2]+=d/2.
			pos=np.array([x.flatten(),y.flatten()])
			if n<np.sqrt(nn):
				for i in range(nn-number):
					extra=np.array([d*(xlen+1+np.floor(i/n)),d_w*(i%n+1)]).reshape(2,1)
					pos=np.hstack([pos,extra])
			orientations=np.random.vonmises(0.0,kappa,nn)
			noise_x=(np.random.random((nn))-np.ones((nn))*0.5)*2.0*noise_int*d
			noise_y=(np.random.random((nn))-np.ones((nn))*0.5)*2.0*noise_int*w
			pos[0]+=noise_x
			pos[1]+=noise_y
			return pos,orientations
		elif state=='milling':
			#print ('generating milling state')
			lower, upper = numberofrings(nn)
			radius=(1.0/2.0+np.arange(upper))*d
			population=np.floor((radius*2.0*np.pi)/d).astype(int)
			totalnumber=np.cumsum(population)
			nr_rings=np.amin(np.where(totalnumber>=nn))+1
			radius=(1./2.+np.arange(nr_rings))*d
			population=np.floor((radius*2.*np.pi)/d).astype(int)
			population[-1]=nn-np.sum(population[:-1])
			distance=2*np.pi*radius/population
			offset=(nr_rings+1)*d
			xpos=[]
			ypos=[]
			orientations=[]
			for i in np.arange(nr_rings):
				theta=2*np.pi*np.linspace(0,1,population[i],endpoint=False)+((np.random.random(population[i])-np.ones(population[i])*0.5)*2.0*noise_int*d)/radius[i]
				orientations.append(theta-np.pi/2.0*np.ones(population[i])+np.random.vonmises(0.0,kappa,population[i]))
				xpos.append(radius[i]*np.cos(theta)+offset)
				ypos.append(radius[i]*np.sin(theta)+offset)
			xpos=np.concatenate(xpos)
			ypos=np.concatenate(ypos)
			orientations=np.concatenate(orientations)
			orientations=cast_to_pm_pi(orientations)
			return np.array([xpos,ypos]),orientations

		else:
			print("state needs to be either milling or grid or dense shifted grid")



	def reset_calculated_variables(self):
		''' Resets all the variables of swarm that are calculated from the original
			input of positions, orientations, ellipse width w and eye position l
		'''
		self.n=len(self.phi)
		self.phi_bounds=np.zeros([2,self.n,self.n])
		self.md_eye=np.zeros([self.n,self.n])
		self.md_center=np.zeros([self.n,self.n])
		self.tp_subj_pol=np.zeros([2,self.n,self.n])
		self.tp_obj_cart=np.zeros([2,self.n,self.n])
		self.segments={}
		self.unclassified=np.ones([self.n,self.n],dtype=bool)
		self.ang_area=np.zeros([self.n,self.n])
		self.ranked_ang_area=np.zeros([self.n,self.n])
		self.p=np.zeros([self.n,self.n]) # network links from rosenthal...
		self.network=nx.DiGraph()
		self.order=-1.
		self.rot_oder=-1.
		self.clustering=np.zeros(self.n)
		self.vis_angles=np.zeros([self.n,self.n])
		self.ranked_vis_angles=np.zeros([self.n,self.n])
		self.eyeinside=()
		self.tp_subj_pos_pred=np.zeros([2,self.n])
		self.tp_obj_cart_pred=np.zeros([2,self.n])
		self.pred_pos=np.zeros(2)
		self.pred_phi=0.
		self.pred_w=1.
		self.pred_bl=1.
		self.pred_segments={}
		self.pos_pred=np.zeros([self.n,2])
		self.occluded=np.zeros(self.n)
		self.vis_edges=[]
		self.visual_field=np.zeros([self.n,self.n,(self.n-1)*2])
	
	def calc_vis_edges(self,res=1000):
		'''calculates the edges of the visual field and the total occluded 
		area for each fish j'''
		vis_field=np.zeros([self.n,res])
		rays=np.linspace(-np.pi,np.pi,res,endpoint=False)
		for i,j in it.product(np.arange(self.n),np.arange(self.n)):
			if i!=j:
				
				lowerbound=((self.phi_bounds[0,i,j]+np.pi)%(2*np.pi))-np.pi
				upperbound=((self.phi_bounds[1,i,j]+np.pi)%(2*np.pi))-np.pi
				hlp1=np.greater_equal(rays,lowerbound)
				hlp2=np.less_equal(rays,upperbound)
				if lowerbound<=upperbound:
					vis=hlp1*hlp2
				else:
					vis=hlp1+hlp2
				vis_field[j,:]+=vis
				#if j==3:
				#	print(i,lowerbound,upperbound,vis)

		binary_vis=vis_field>0
		binary_vis=binary_vis.astype(int)
		self.occluded=np.sum(binary_vis,axis=1).astype(float)/res #ratio of occluded to free rays
		self.vis_edges=abs(np.roll(binary_vis,1)-binary_vis)*(np.repeat([rays-np.pi*2./res],self.n,axis=0))

	def calc_visfield(self,check_intersects=True,fake_3D=False,reposition=True):
		# initialize class properties (used as global variables)
		self.reset_calculated_variables()
		# calculate metric distances between ellipses
		self.calc_md()
		# calculate tangent points (=outmost points on ellipses in visual field of other ellipses)
		inters=self.calc_tps(check_intersects=check_intersects,fake_3D=fake_3D)
		print(inters)
		if inters!=-1 and reposition:
			print('moving ellipses to get rid of intersections')
			self.remove_intersections()
			self.calc_md()
			inters=self.calc_tps(check_intersects=check_intersects,fake_3D=fake_3D)
			print(inters)
		if inters==-1:
			#self.calc_vis_angle()
			#self.rank_vis_angles()
			self.calc_vis_field_and_ang_area()
			#self.calc_vis_segments()
			#self.calc_ang_area()
			return 1
		else:
			print('FAILED TO REMOVE INTERSECTIONS!')
			return 0
				
	def calc_binary_visual_network(self,abs_vis_threshold=0.,rel_vis_thresh=0.,create_networkX=False):
		self.calc_visfield()
		visibility=self.ang_area>0.
		mask1=self.ang_area>=abs_vis_threshold
		mask2=self.ang_area/self.vis_angles>=rel_vis_thresh
		self.p=np.array(visibility*mask1*mask2,dtype=int)
		if create_networkX:
			self.create_network(allinfo=True)
	

	def calc_adjmat(self,beta1,beta2,beta3,ang_area_threshold,dist_threshold,logbase,check_intersects=True,visreq=True):
		self.reset_calculated_variables()
		self.calc_md()
		inters=self.calc_tps(check_intersects=check_intersects,fake_3D=not(visreq))
		if inters==-1:
			self.calc_vis_angle()
			self.rank_vis_angles()
			self.calc_vis_segments()
			self.calc_ang_area()
			self.rank_ang_area(visibility_required=visreq)
			self.calc_p(beta1,beta2,beta3,logbase=logbase,ang_area_threshold=ang_area_threshold,dist_threshold=dist_threshold,use_vis_angles=not(visreq))
			return 1
		else:
			print('Intersections found:\n',inters)
			return 0

	def get_adjmat(self):
		return self.p

	def calc_md(self):
		'''
 		calculates the euclidean distance between all ellipse:
		-(for the eyes of the ellipses) and saves it to self.md_eye
		-(for the geometric center of the ellipse) and saves it to self.md_center
		'''
		z=np.array([[complex(p[0],p[1]) for p in self.pos.T]])
		z_center=np.array([[complex(p[0],p[1]) for p in self.pos_center.T]])
		self.md_eye=abs(z.T-z)
		self.md_center=abs(z_center.T-z_center)
		self.md_eye_to_center=abs(z_center.T-z) 
	
	def check_intersects(self):
		''' returns -1 if there are no intersections between any of the ellipses in the swarm
			if intersections are presents a list of ids of intersecting ellilpses is returned
			together with a list of ids where the eye of the second entry is in the body of the
			 first entry'''
		md_center=self.md_center
		intersecting=False
		possible_intersect=np.array([np.array([a,b]) for a in range(self.n) for b in range(a) if md_center[a,b]<1.]).T
		intersect_list=[]
		eye_inside_list=[]
		if np.sum(possible_intersect)!=0:
			'''check if canditates actually intersect and raise error if intersection is found '''
			for i in range(len(possible_intersect[0])):
				test=self.ellipse_intersect(possible_intersect[0,i],possible_intersect[1,i])
				if test[0]:
					intersect_list.append(list(possible_intersect[:,i]))
					intersecting=True
					if test[1]==True:
						eye_inside_list.append(list(possible_intersect[:,i]))
					if test[2]==True:
						eye_inside_list.append(list(np.flip(possible_intersect[:,i])))
		if intersecting:
			return [intersect_list,eye_inside_list]
		else:
			return [-1,-1]
		

	def ellipse_intersect(self,id1,id2):
		'''determines if ellipse with id1 and ellipse with id2 are intersecting
		returns boolean array with 3 entries:
			0: Do the ellipses intersect? (True/False)
			1: is the eye of ellipses id2 inside the body of ellipse id1?
			2: is the eye inside the other way around?
		'''
		phi1=self.phi[id1]
		phi2=self.phi[id2]
		pos1=self.pos_center[:,id1]
		pos2=self.pos_center[:,id2]
		w=self.w 
		pos1_eye=self.pos[:,id1]
		pos2_eye=self.pos[:,id2]

		ellipses = [(pos1[0], pos1[1], 1, w/2.0, phi1), (pos2[0], pos2[1], 1, w/2.0, phi2)]
		a, b =ellipse_polyline(ellipses)
		inter=intersections(a,b)
		if inter:
			two_in_one=point_inside(pos2_eye,a)
			one_in_two=point_inside(pos1_eye,b)	
			return np.array([True,two_in_one,one_in_two])
		else:	
			return np.array([False,False,False])



	def calc_tps(self,check_intersects=True,fake_3D=False):
		''' calculates the tangent points (tps) of ellipses after checking that they don't overlap
		'''

		'''#############  check for intersecting ellipses  ######################'''		
		'''identify possible candidate pairs by their distance'''

		if not check_intersects:
			print("You are not checking for intersections of the ellipses. In case an eye of one ellipses\
				lies inside the body of another ellipses, the analytical calculation will not work\
				and you will get an error. ")
			if fake_3D:
				print("Creating an artificial 3rd dimension by setting the angular area of ellipses \
					that contain the eye of another ellipse to pi, will not work unless you set\
					check_intersects=True. Defaulting back to fake_3d=False.")
				fake_3D=False
		check1=-1
		check2=-1
		if check_intersects:				
			check1,check2=self.check_intersects()
			
			
		if not fake_3D and check1!=-1:
			print('remove_intersections() needed')
			#	print('Intersection of ellipses detected. Please use remove_intersections() to eliminate these and try calc_tps again afterwards. You can also use fake_3D=True to set visual angle to 180 degree for ellipse A seeing B if the eye of A is inside B. In this case B is not said to block the view of A. Creating of visual field might currently not work for the fake_3D option because neighbors might have been assumed to not overlap.')
		
		else:
			if check2!=-1:
				eye_inside=tuple([list(a) for a in np.array(check2).T])
			'''###########   calculate tangent points  #########################'''
			z=np.array([[complex(p[0],p[1]) for p in self.pos.T]])
			z_center=np.array([[complex(p[0],p[1]) for p in self.pos_center.T]])
			r=abs(z_center.T-z)
			#indices ij: abs(z_center(i)-z(j)), j is observer, i target
	
			'''to avoid errors in further calc. result for these will be set manually'''
			np.fill_diagonal(self.md_center,float('NaN'))
			np.fill_diagonal(self.md_eye,float('NaN'))
			np.fill_diagonal(r,float("NaN"))
		
			if isinstance(check2,list):
				if len(check2)!=0:
					r[eye_inside]=float("NaN")
				check1=-1
			'''initialize variables'''
			w=self.w
			phi_m=np.array([self.phi,]*self.n).transpose()
			tp_subj=[]
			tp_obj=[]
			pt_subj=np.zeros(2)
			pt_obj=np.zeros(2)
			theta_tp=0.0
			r_tp=0.0
			x=self.pos[0]
			y=self.pos[1]
			x_center=self.pos_center[0]
			y_center=self.pos_center[1]
			rel_x=x_center.reshape(len(x_center),1)-x #entry(ij)=pos(i)-pos(j)
			rel_y=y_center.reshape(len(y_center),1)-y
	
			'''relative position of i to j'''
			theta=np.arctan2(rel_y,rel_x)
			'''calculate tangent points' parameter psi in parametric ellipse eq.'''
			psi=psi_go(w,r,theta,phi_m)
			for p in psi:
				'''calculate tangent point from psi in local polar coordinates'''
				pt_subj=ellipsepoints_forgo(r,theta,phi_m,p,w)
				z_pt_subj=pt_subj[0]+1j*pt_subj[1]
				theta_tp=cast_to_pm_pi(np.arctan2(pt_subj[1],pt_subj[0])-self.phi)
				r_tp=abs(z_pt_subj)
				np.fill_diagonal(r_tp,0.0)
				tp_subj.append(np.array([r_tp,theta_tp]))
				'''transform tp to cartesian global coordinates'''
				pt_obj=pt_subj+np.array([np.array([self.pos[0],]*self.n),np.array(\
				[self.pos[1],]*self.n)])
				np.fill_diagonal(pt_obj[0],0.0)
				np.fill_diagonal(pt_obj[1],0.0)
				tp_obj.append(pt_obj)
			self.tp_subj_pol=np.array(tp_subj)
			self.tp_obj_cart=np.array(tp_obj)
			
			if check2!=-1:
				self.eyeinside=tuple(eye_inside)

		return check1

	def calc_vis_angle(self):

		looplist=it.product(range(self.n),range(self.n))
		'''save upper and lower limit of angular area subtended by fish i on
		the eye of fish j, with phi_bounds[0,i,j] <=phi_bounds[1,i,j]'''
		for observer,target in looplist:
				vis_angle= smallestSignedAngleBetween(self.tp_subj_pol[0,1,target,observer]\
					,self.tp_subj_pol[1,1,target,observer])
				if vis_angle>=0:
					 self.phi_bounds[:,target,observer]=self.tp_subj_pol[:,1,target,\
					observer]
				else:
					self.phi_bounds[:,target,observer]=self.tp_subj_pol[::-1,1,\
					target,observer]
					'''manually set the boundaries for the area subtended by i on eye of i
					to -pi,-pi, resulting in an area of 0'''
				self.vis_angles[target,observer]=vis_angle
		np.fill_diagonal(self.phi_bounds[0,:,:],-np.pi)
		np.fill_diagonal(self.phi_bounds[1,:,:],-np.pi)
		if len(self.eyeinside)!=0:
			'''manually setting the angular area of those with eyeinside to pi'''
			self.phi_bounds[0][self.eyeinside]=0.0
			self.phi_bounds[1][self.eyeinside]=np.pi
			self.vis_angles[self.eyeinside]=np.pi
		np.fill_diagonal(self.vis_angles,0) 

	def calc_only_vis_angle(self):
		looplist=it.product(range(self.n),range(self.n))
		for observer,target in looplist:
			self.vis_angles[target,observer]=smallestSignedAngleBetween(self.tp_subj_pol[0,1,target,observer],self.tp_subj_pol[1,1,target,observer])	
		np.fill_diagonal(self.vis_angles,0)



	def remove_intersections(self,fileName='random',lamda1=0.05, overdamp=0.5):
		# Lamda1 - coefficient of the repulsion area of the cells (their main body) (0.01 - 0.05) \n");
       		# overdamp - coeffiecient that controls cell inertia (0 - 1).\n");

		'''This function uses C++-code to shift and turn the ellipses such that they don't intersect anymore'''
		
		# save the position data to file
		if fileName=='random':
			fileName=str(int(np.random.random()*1000000))
		outpath='./position_orientation_data_tmp/'+fileName
		pospath=outpath+'_pos.txt'
		headingpath=outpath+'_phi.txt'
		np.savetxt(pospath,self.pos_center.T,fmt='%1.8f')
		np.savetxt(headingpath,self.phi,fmt='%1.8f')
		
		resultpath=outpath
		if not os.path.exists(resultpath):
			os.makedirs(resultpath)
		#print(resultpath)
		#print("LD_LIBRARY_PATH=$HOME/lib ./Palachanis2015_particle_system/build/pS 50000 {} {} {} 0.05 0.2 {} {} {} {} {} 1 0.5".format(180,self.n,self.w,pospath,headingpath,resultpath,1.,1.))
		command="LD_LIBRARY_PATH=$HOME/lib ./Palachanis2015_particle_system/build/pS 50000 {} {} {} {} {} {} {} {} 1.06 1. 0.".format(180,self.n,(self.w+0.06)/1.06,lamda1,overdamp,pospath,headingpath,resultpath)
		os.system(command)
		new_pos=np.loadtxt(resultpath+'/pos_d1.000_w%1.2f_bl1.1.txt'%((self.w+0.06)/1.06))
		hxhy=np.loadtxt(resultpath+'/headings_d1.000_w%1.2f_bl1.1.txt'%((self.w+0.06)/1.06))
		#print(np.shape(hxhy))
		new_phi=np.arctan2(hxhy[:,1],hxhy[:,0])
		self.set_pos_orient(new_pos,new_phi,center=True)
		shutil.rmtree(resultpath)
		os.remove(pospath)
		os.remove(headingpath)

	def get_ranked_tangentpoints(self,nr):
		hlp=np.zeros((2*self.n,6))  #r,phi,idnr(ellipse),id ranked distance,id ranked phi, mindist(tp1,tp2,center)
		tmp=self.tp_subj_pol[:,:,:,nr]
		for i in range(self.n):			
			#first tangent poin
			hlp[2*i,2]=int(i)
			hlp[2*i,1]=tmp[0,1,i]
			hlp[2*i,0]=tmp[0,0,i]
			#second tangent point
			hlp[2*i+1,2]=int(i)
			hlp[2*i+1,1]=tmp[1,1,i]
			hlp[2*i+1,0]=tmp[1,0,i]
		count=0
		sort=np.array(sorted(hlp,key=getKey0))
		for i in np.arange(0,2*self.n,1):
			sort[i,3]=int(i)        #ranks all tp
		sort=np.array(sorted(sort,key=getKey1))
		for i in np.arange(0,2*self.n,1):
			sort[i,4]=int(i)    #ranks all tp,
			distcenter=self.md_eye_to_center[int(sort[i,2]),nr]
			sort[i,5]=min(tmp[0,0,int(sort[i,2])],tmp[1,0,int(sort[i,2])],distcenter)
		return sort

	def calc_vis_field_and_ang_area(self):

		'''1. Calculates the visual field for each ellipse and saves it to 
		   self.visual_field, an nxnx2(n-1) array, indices ijk as follows:
		   i: id of ellipse visible/lower boundary/upper boundary
		   j: viewer id
		   k: which section of visual field
		   (a np.nan entry means no occlusion of visual field in this area)
		   2. then calculates the angular area of each ellipse in the visual field of all 
		   other ellipses and saves it to self.ang_area, a numpy nxn array 
		   indices ij:
		   i: seen individual (the one who is seen by individual j)
		   j: focal individual (the one whose visual field is given)'''

		# get ray angles for each ellipse
		angles=self.tp_subj_pol[:,1].flatten(order='f')
		angles=np.sort(angles[~np.isnan(angles)].reshape(2*(self.n-1),self.n,order='f').T)
		assert np.logical_and(angles.all()<=np.pi, angles.all()>=-np.pi), 'angles are not in pm pi interval'
		between_angles=cast_to_pm_pi(np.diff(angles,append=(2.*np.pi+angles[:,0]).reshape(self.n,1),axis=1)/2.+angles)

		#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#  transformation of angles for the calculation of intersection points
		#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		# transform the local angles into points in global cartesian coordinates
		phi=self.phi
		phi_hlp=np.repeat(phi.reshape(self.n,1),2*(self.n-1),axis=1)
		transf_betw_ang=between_angles+phi_hlp
		raypoints=np.array([np.cos(transf_betw_ang),np.sin(transf_betw_ang)])+np.tile(self.pos,((self.n-1)*2,1,1)).transpose(1,2,0)

		# here we need to transform the raypoints from global coordinates to local 
		# ones of the ellipse that we want to check of intersections 
		# (in a manner that will set up a nested for loop)
		raypoints=np.tile(raypoints,(self.n,1,1,1)).transpose(1,0,2,3) 
		#indices: x/y ,N repetitions (in which coordinate system),focalid (seen from which eye),raypoints (which tangent point)

		pos_hlp=np.tile(self.pos_center,(2*(self.n-1),1,1)).transpose(1,2,0)
		pos_hlp=np.tile(pos_hlp,(self.n,1,1,1)).transpose(1,2,0,3)
		#indices: ijkl x/y,id (coordinate syst.=the individual that intersections will be found for), repetition (which eye), repetitions (which tangent point)
		# shifting the raypoints to a coordinate system with origin in the center of the ellipse j (the one that intersections will be found for)
		raypoints-=pos_hlp

		#now go to polar coordinates and rotate the points by -phi, 
		# to orient the ellipse j along positive x-axis in the respective
		# coordinate system (this is needed because the function calculating 
		# intersections assumes an ellipse at the center with this orientation)
		r=np.sqrt(raypoints[0]**2+raypoints[1]**2)
		theta=np.arctan2(raypoints[1],raypoints[0])
		phi_hlp=np.tile(phi,(self.n,(self.n-1)*2,1)).transpose(2,0,1)
		theta-=phi_hlp
		# now the transofmration is over
		raypoints=np.array([r*np.cos(theta),r*np.sin(theta)])


		# Now we need to similarly transform the eye positions from 
		# global to local (in a manner that will set up a nested for loop)
		# (the id of the viewer ellipse is the second last index, thus 
		# the array needs to have repetitions for all other axes)
		eyes=np.tile(self.pos,(2*(self.n-1),1,1)).transpose(1,2,0)
		eyes=np.tile(eyes,(self.n,1,1,1)).transpose(1,0,2,3)
		#shift coordinate system origins
		eyes-=pos_hlp
		#rotate coordinate systems
		r=np.sqrt(eyes[0]**2+eyes[1]**2)
		theta=np.arctan2(eyes[1],eyes[0])
		theta-=phi_hlp
		eyes=np.array([r*np.cos(theta),r*np.sin(theta)])
		#transformation done
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++


		#         Calculation of intersection points            
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
		inters=get_ellipse_line_intersection_points(eyes,raypoints,self.w)
		inters=remove_self_intersections(inters,self.n)
		# indices: [x/y, which intersection, on which ellipse, 
		# for which viewer, for which ray]
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

		# all intersection points are still in coordinates of 
		# the 'on which ellipse' ellipse, transform to global coordinates next:
		#1. rotate by +phi 
		theta=np.arctan2(inters[1],inters[0])+phi_hlp     
		r=np.sqrt(inters[0]**2+inters[1]**2)
		inters=np.array([r*np.cos(theta),r*np.sin(theta)])
		# 2. and shift position of origin
		pos_hlp=np.tile(pos_hlp,(2,1,1,1,1)).transpose(1,0,2,3,4)
		inters=inters+pos_hlp

		# in order to decide which intersection point is closest to an 
		# ellipse we need to move to the coordinate system of the ellipse 
		# which is emitting the rays from its eye (second last index)
		# (we skip the rotation because we are only interested in the
		# distances r anyways)
		pos_hlp=np.tile(self.pos,(2*(self.n-1),1,1)).transpose(1,2,0)
		pos_hlp=np.tile(pos_hlp,(self.n,1,1,1)).transpose(1,2,0,3)
		pos_hlp=np.tile(pos_hlp,(2,1,1,1,1)).transpose(1,0,3,2,4)
		#shift to the local coordinates
		inters-=pos_hlp
		#calculate the distances:
		r=np.sqrt(inters[0]**2+inters[1]**2)

		#Here want to find for each ray emitted from the eye of a viewer ellipse, 
		# the id of the closest ellipse it intersects with
		out=np.empty([self.n,(self.n-1)*2],dtype=float)
		closest_id=get_closest_id(r,out,self.n)

		self.visual_field=np.stack([closest_id,angles,np.roll(angles,-1,axis=-1)]) 
		# 1st index: id of ellipse visible/lower boundary/upper boundary
		# 2nd index: viewer id
		# 3rd index: which section of visual field

		area=np.stack([closest_id,(np.diff(self.visual_field[1::,:,:],axis=0)%np.pi)[0]])
		# id and area for each section of visual field of each ellipse
		# indices ijk:
		# i: id/angle
		# j: viewer id
		# k: section id

		# calculate angular area:
		angular_area=np.zeros([self.n,self.n],dtype=float)
		for i in range(self.n):
		    mask=area[0]==i
		    angular_area[i,:]=np.sum(mask*area[1],axis=-1)
		self.ang_area=angular_area




	def calc_vis_segments(self,fake_3D=False):
		'''####################################################################
		figures out the upper and lower boundary of the area subtended by 
		ellipse i on the eye of ellipse j (multiple boundaries are given if
		splitting occurs through partial occlusion)
		saves these boundaries to self.segments
		####################################################################'''
		if not fake_3D:
			n=self.n
			splitting={}
			phi_b=np.zeros([2,n])
			'''notchecked_array (i,j): if False, the ellipse i is not considered as
			possibly occluding any other ellipse from the view of j, this is used
			to avoid checking every ellipse in the cone of i twice (because of 
			its two tanget points, see below)'''
			notchecked_array=np.ones([n,n],dtype=bool)

			for k in range(n):#Index of the observer
				#print('........................')
				#print('viewer:',k)
				phi_b=np.array(self.phi_bounds[:,:,k])
				#print('viewing object boundaries at:\n',phi_b)
				'''obtain tangent points sorted by their angular position phi in 
				the view of k'''
				tp_ranked_phi=self.get_ranked_tangentpoints(k)
				#print('tp_ranked_phi\n',tp_ranked_phi)
				'''obtain tangent points sorted by their distance to the eye of k'''
				tp_ranked_r=np.array(sorted(tp_ranked_phi,key=getKey0))
				tp_ranked_mindist=np.array(sorted(tp_ranked_phi,key=getKey5))
				'''go through the tangent points in the view of k, starting with 
				the closest'''
				for tp in tp_ranked_mindist: #all targets' tangent points
					'''if the target isn't the observer and hasn't been classified,
					that is, marked as completely invisible or has a fixed value or
					 already was considered as target or marked completely visible
					'''
				#	print('tangent point',tp)
					if tp[2]!=k and self.unclassified[int(tp[2]),k]:
					#	if k==73:
					#		print('106',self.phi_bounds[:,106,73],'\n46',self.phi_bounds[:,46,73],'\n57',self.phi_bounds[:,57,73],'\n91',self.phi_bounds[:,91,73],'\n6',self.phi_bounds[:,6,73])
					#		print('106',phi_b[:,106],'46',phi_b[:,46],'57',phi_b[:,57],'91',phi_b[:,91],'6',phi_b[:,6])
					#		print('106',self.unclassified[106,73],'\n46',self.unclassified[46,73],'\n57',self.unclassified[57,73],'\n91',self.unclassified[91,73],'\n6',self.unclassified[6,73])
					#		print('checking '+str(tp[2])+' now')
				#		print('classifying tp')
						self.unclassified[int(tp[2]),k]=False
						'''check if there are tangent points with angular positions
						between the two tangent points of ellipse with id tp[2], 
						that is, if there are any ellipses in the cone of tp[2]'''
						partner_id=get_partnerid(tp,tp_ranked_r)
				#		print('partnerid',partner_id)
						sig=int(np.sign(smallestSignedAngleBetween(tp[1],\
							tp_ranked_r[partner_id,1])))
				#		print('sig',sig)
						tmp=2*n
						length=(sig*(tp_ranked_r[partner_id,4]-tp[4])-1)%tmp
				#		print('length:',length)
						if length!=0:
							if sig==1:
								 l=np.arange(0,2*n,1)
							else:
								l=np.arange(0,2*n,1)[::-1]
				#			print('l',l)
							
							inbetween=it.islice(it.dropwhile(lambda x: x != \
									int((tp[4]+sig)%tmp),it.cycle(l)),None,int(length))
				#			print('inbetween',inbetween)
							notchecked=np.ones(2*n,dtype=bool)
							#print np.ones([n,n],dtype=bool)[self.intersections]
							notchecked=notchecked_array[k]
							notchecked[k]=False
							'''for all the tps in the cone of ellipse tp[2]:'''
							for nr in inbetween:
								#if k==73:
							#		print('nodes inbetween:',inbetween)
								'''to avoid checking an ellipse twice if both its tangent
								points lie in inbetween'''
								if notchecked[int(tp_ranked_phi[nr,2])]:
									#notchecked[int(tp_ranked_phi[nr,2])]=False
									#print 'nr '+str(nr)+' belonging to el. '+str(tp_ranked_phi[nr,2])

									phi_partner=tp_ranked_phi[get_partnerid(\
										tp_ranked_phi[nr],tp_ranked_phi),1]
									''' if the partner_tp of the tp_ranked_phi
									[nr] is also in the cone'''
									if smallestSignedAngleBetween(tp_ranked_r[partner_id,1]\
										,phi_partner)*smallestSignedAngleBetween(tp[1],phi_partner)<0\
										and abs(smallestSignedAngleBetween(tp_ranked_r[partner_id,1]\
										,phi_partner))+abs(smallestSignedAngleBetween(tp[1],phi_partner))<np.pi:
										#if k==73:
					#						print('both tps in cone')
										'''if ellipse[nr] is closer to observer it
										splits the visible area of the target, the 
										occluded interval is saved in splitting var'''
										if is_closer([tp_ranked_phi[nr,0],tp_ranked_phi[nr,1]],\
											[tp_ranked_r[partner_id,0],tp_ranked_r[partner_id,1]],\
											[tp[0],tp[1]]):
										#	if k==73:
					#							print('ellipse '+str(int(tp_ranked_phi[nr,2]))+' is closer than '+str(tp[2]))
											if not (int(tp[2]),k) in splitting:
												splitting[int(tp[2]),k]=[]
													
											if smallestSignedAngleBetween(tp_ranked_phi[nr,1],phi_partner)>0:
												splitting[int(tp[2]),k].append([tp_ranked_phi[nr,1],phi_partner])
											else:
												splitting[int(tp[2]),k].append([phi_partner,tp_ranked_phi[nr,1]])

										else:
											'''if it is further away, it is completely
											occluded by the target and can be skipped in
											the for tp in tp_ranked_r loop for this 
											observer k, also if it was split is unimpor-
											tant and can be deleted'''
											if self.unclassified[int(tp_ranked_phi[nr,2]),k]:
												self.unclassified[int(tp_ranked_phi[nr,2]),k]=False
												phi_b[:,int(tp_ranked_phi[nr,2])]=np.zeros(2)
												if (int(tp_ranked_phi[nr,2]),k) in splitting:
													del splitting[int(tp_ranked_phi[nr,2]),k]
												#if k==73:
					#							#	print(str(tp[2])+' completely blocks '+str(int(tp_ranked_phi[nr,2])))	
									else:
										'''if the ellipse only partly lies in the 
										cone of the target'''
					#					if k==73:
					#						print(str(int(tp_ranked_phi[nr,2]))+' lies partly in cone of '+str(tp[2]))
										if is_closer([tp_ranked_phi[nr,0],tp_ranked_phi[nr,1]],\
											[tp_ranked_r[partner_id,0],tp_ranked_r[partner_id,1]],[tp[0],tp[1]]):
											'''if target is occluded by this ellipse
											reset the out boundaries of the visible
											ang area of the target'''
					#						if k==73:
					#							print(str(int(tp_ranked_phi[nr,2]))+' partly occludes '+str(tp[2]))
											if smallestSignedAngleBetween(tp_ranked_phi[nr,1],phi_partner)>=0:
												hidden=[tp_ranked_phi[nr,1],phi_partner]
											else:
												hidden=[phi_partner,tp_ranked_phi[nr,1]]
											visible=[phi_b[0,int(tp[2])],phi_b[1,int(tp[2])]]
											mer=merge_segments(visible,hidden)
											if mer!=[]:
												phi_b[:,int(tp[2])]=np.array(mer[0])
											else:
												phi_b[:,int(tp[2])]=np.zeros(2)
												#self.unclassified[int(tp[2]),k]=False
										else:
											'''if the target is occluding the other
											ellipse, reset the other ellipse's visible
											area's boundaries'''
											#print(str(nr)+' is partly occluded by '+str(tp[2]))
											visible=[phi_b[0,int(tp_ranked_phi[nr,2])],phi_b[1,int(tp_ranked_phi[nr,2])]]
											if smallestSignedAngleBetween(tp_ranked_r[partner_id,1],tp[1])>=0:
												hidden=[tp_ranked_r[partner_id,1],tp[1]]
											else:
												hidden=[tp[1],tp_ranked_r[partner_id,1]]
											mer=merge_segments(visible,hidden)
											if mer!=[]:
												phi_b[:,int(tp_ranked_phi[nr,2])]=np.array(mer[0])
											else:
												phi_b[:,int(tp_ranked_phi[nr,2])]=np.zeros(2)
												#self.unclassified[int(tp_ranked_phi[nr,2]),k]=False
						else:
							'''if there are no tangent points in the cone'''
							self.unclassified[int(tp[2]),k]=False
				included=np.zeros(2,dtype=bool)
				'''reset the upper and lower boundary of the visible areas in the
				visual field of ellipse k to those determined in the process above'''
				self.phi_bounds[:,:,k]=phi_b
			#self.ang_area=(self.phi_bounds[1,:,:]-self.phi_bounds[0,:,:])%np.pi
			'''join the info about the outer visible boundaries of each ellipse with 
			the info about occluded areas that split the visible area of a target'''
			for j,k in it.product(range(n),range(n)):
				self.segments[j,k]=[self.phi_bounds[:,j,k]]

			for key,splits in splitting.items():
				segments=[self.phi_bounds[:,key[0],key[1]]]
				for hidden in splits:
					seg_list=[]
					for visible in segments:
						seg_list.extend(merge_segments(visible,hidden))
					segments=seg_list
				self.segments[key[0],key[1]]=segments
		else:
			for j,k in it.product(range(self.n),range(self.n)):
                                self.segments[j,k]=[self.phi_bounds[:,j,k]]
		#	hlp=self.phi_bounds
	#		self.segments={(i,j):hlp[:,i,j] for i in range(self.n) for j in range(self.n)}
	def calc_ang_area(self):
		n=self.n
		ang_area=np.zeros([n,n])
		for j,k in it.product(range(n),range(n)):
			for seg in self.segments[j,k]:
				ang_area[j,k]+=(seg[1]-seg[0])%np.pi
		self.ang_area=ang_area

	def rank_ang_area(self,visibility_required=True):
		n=self.n
		ranked_ang_area=np.zeros([n,n])
		for jk in range(n):
			hlp=np.array(self.ang_area[:,jk])
			ranking=n*np.ones(n)-np.argsort(np.argsort(hlp))
			ranked_ang_area[:,jk]=ranking
			if visibility_required:
				ind=np.where(hlp==0.0)
				ranked_ang_area[ind,jk]=-1
			else:
				ind=np.where(hlp==0.0)
				ranked_ang_area[ind,jk]=n
		self.ranked_ang_area = ranked_ang_area

	def rank_vis_angles(self):
		n=self.n
		ranked_vis_angles=np.zeros([n,n])
		for jk in range(n):
	#		print(jk, 'is viewer')
			hlp=np.array(self.vis_angles[:,jk])
			hlp1=hlp[hlp<np.pi]
#			print(hlp1)
			n1=n-len(hlp1)
			ranking=(n1+len(hlp1))*np.ones(len(hlp1))-np.argsort(np.argsort(hlp1))
#			print(ranking)
			hlp2=np.ones_like(hlp)
			hlp2[hlp<np.pi]=ranking#+len(hlp1)
			ranked_vis_angles[:,jk]=hlp2
		self.ranked_vis_angles = ranked_vis_angles


	def calc_p(self, beta1, beta2, beta3, logbase='10', ang_area_threshold=0.02, dist_threshold=10.,use_vis_angles=False):
		md=np.copy(self.md_center)
		rank_area=self.ranked_ang_area
		np.fill_diagonal(md,np.inf)
		mask1=self.ang_area>ang_area_threshold
		mask2=md<dist_threshold
		mask=mask1*mask2
		raa=self.ranked_ang_area
		if use_vis_angles:
			raa=self.ranked_vis_angles
		if logbase=='10':#this is needed for the Matt data coefficients
			p=mask/(1.0+np.exp(-beta1-beta2*np.log10(self.md_center*self.bl)-beta3*raa))
		else: #this is need for the Rosenthal data coefficients
			p=mask/(1.0+np.exp(-beta1-beta2*np.log(self.md_center*self.bl)-beta3*raa))
		ind=np.where(rank_area==-1)
	#	print(ind)
#		print('indices')
		p[ind]=0.0
		np.fill_diagonal(p,0.0)
#		print(p)
		self.p=p


	def order_func(self):
		out=np.sqrt(np.sum(np.cos(self.phi))**2+\
			np.sum(np.sin(self.phi))**2)/float(self.n)
		self.order=out
		return out


	def rot_order(self):
		center=np.sum(self.pos,axis=1)/self.n
		rel_pos=np.array([self.pos[0]-center[0],self.pos[1]-center[1]])
		velocity=np.array([np.cos(self.phi),np.sin(self.phi)])
		rot_order=np.sum(abs(np.cross(rel_pos.T,velocity.T))/(np.linalg.norm(rel_pos.T,axis=1)*np.linalg.norm(velocity.T,axis=1)))/self.n
		if rot_order>1 or rot_order<0:
			print('rot order not in range')
		self.rot_order=rot_order
		return rot_order


	def CalcWeightedClusteringCoeff(self,weightThreshold=0.0):
		''' Function by Bryan'''
		''' Calc local weighted clustering coefficient as given 
		in Eq. S11 in Rosenthal and Twomeyet al. 
		'Simple' implementation - slow - maybe can be improved.
		Input: 
		------
		weightsMatrix - weight matrix as obtained from the matlab file (field 'W')
		weightThreshold - threshold for binarizing the adjecency matrix
		'''
		dot = np.dot
		w = np.array(self.p)
		w_sym = w + w.T
		a = w > weightThreshold
		d_tot = np.sum(a,axis=0)+ np.sum(a,axis=1).T
		d_sym = np.sum(a*a.T,axis=0)
		#b=np.array(a,dtype=int)
		#d_sym1 = np.diag(dot(b,b))  
		div = 2.*( d_tot*(d_tot-1.) - 2.*d_sym )
		clustering=np.diag(dot(dot(w_sym,w_sym),w_sym) ) / div
		self.clustering=clustering
		return clustering

	def detect_predator(self,pred_pos,pred_phi,pred_bl,w_predator=1.,rate=1.):
		r=np.sqrt((self.pos[0]-pred_pos[0])**2+(self.pos[1]-pred_pos[1])**2)
		self.pos_pred[:,0]=r
		w=w_predator
		phi_m=pred_phi*np.ones(self.n)
		tp_subj=[]
		tp_obj=[]
		pt_subj=np.zeros(2)
		pt_obj=np.zeros(2)
		theta_tp=0.0
		r_tp=0.0
		x=self.pos[0]
		y=self.pos[1]
		x_center=self.pos_center[0]
		y_center=self.pos_center[1]
		rel_x=-self.pos[0]+pred_pos[0]
		rel_y=-self.pos[1]+pred_pos[1]
		self.pred_pos=pred_pos
		self.pred_phi=pred_phi	
		self.pred_w=w_predator
		self.pred_bl=pred_bl
		'''relative position of i to j'''
		theta=np.arctan2(rel_y,rel_x)
		self.pos_pred[:,1]=theta
		'''calculate tangent points' parameter psi in parametric ellipse eq.'''
			
		psi=psi_go(w,r,theta,phi_m,l=pred_bl/2.)
		for p in psi:
			'''calculate tangent point from psi in local polar coordinates'''
			pt_subj=ellipsepoints_forgo(r,theta,phi_m,p,w,l=pred_bl/2.)
			z_pt_subj=pt_subj[0]+1j*pt_subj[1]
			theta_tp=cast_to_pm_pi(np.arctan2(pt_subj[1],pt_subj[0])-self.phi)
			r_tp=abs(z_pt_subj)
			tp_subj.append(np.array([r_tp,theta_tp]))
			'''transform tp to cartesian global coordinates'''
			pt_obj=pt_subj+self.pos
			tp_obj.append(pt_obj)
		self.tp_subj_pol_pred=np.array(tp_subj)
		self.tp_obj_cart_pred=np.array(tp_obj)

		for i in range(self.n):
			#get the visual angle of the predator for fish i
			tp1=self.tp_subj_pol_pred[0,1,i]
			tp2=self.tp_subj_pol_pred[1,1,i]
			if smallestSignedAngleBetween(tp1,tp2)>=0:
				predator=[[tp1,tp2]]
			else:
				predator=[[tp2,tp1]]
			# now go through all the other fish j!=i and 
			# check if they occlude the predator for fish i
			j=0
			for j in range(self.n):
				new_predator=[]
				# get the visible range of fish j
				fish_j_tp1=self.tp_subj_pol[0,1,j,i]
				fish_j_tp2=self.tp_subj_pol[1,1,j,i]
				if np.isnan(fish_j_tp1):
					fish_j_tp1=0.
				if np.isnan(fish_j_tp2):
					fish_j_tp2=0.
				if smallestSignedAngleBetween(tp1,tp2)>=0:
					fish_j=[fish_j_tp1,fish_j_tp2]
				else:
					fish_j=[fish_j_tp2,fish_j_tp1]   
				#check if the fish_j occludes the predator (the predator may also be a list of segments
				# once we get to the second fish we are checking, so we check each segment (pred) of the predator )
				for pred in predator:
					if pred!=[]:
						if np.diff(fish_j)!=0:
							if np.sum(np.sign(pred))==-np.sum(np.sign(fish_j)) and np.sum(np.sign(pred))!=0:
								ret=[pred]
							else:
								ret=merge_segments(pred,fish_j)
						else:
							ret=[pred]
						# add the visible segment to the list for the updated visibility of predator
						new_predator+=ret
					predator=new_predator
				self.pred_segments[i]=predator
		   




	def plot_predator(self,ax,color='k',alpha=1.,zorder=10,plot_detection_radius=False,plot_pred_dist=False):
		if plot_detection_radius:
			zone=Ellipse(self.pred_pos,plot_detection_radius*2.,plot_detection_radius*2.,0.)
			ax.add_artist(zone)
			zone.set_clip_box(ax.bbox)
			zone.set_facecolor('none')
			zone.set_edgecolor('k')
			zone.set_linewidth(1)
		if plot_detection_radius:
			zone=Ellipse(self.pred_pos,plot_pred_dist*2.,plot_pred_dist*2.,0.)
			ax.add_artist(zone)
			zone.set_clip_box(ax.bbox)
			zone.set_facecolor('none')
			zone.set_edgecolor('k')
			zone.set_linewidth(0.2)
			zone.set_linestyle('--')

		ellipse=Ellipse(self.pred_pos,self.pred_w*self.pred_bl,self.pred_bl,self.pred_phi*180.0/np.pi-90.)
		ax.add_artist(ellipse)
		ellipse.set_clip_box(ax.bbox)
		ellipse.set_facecolor(color)
		ellipse.set_alpha(alpha)
		ellipse.set_zorder(zorder)


	def plot_predator_visibility(self,ax,focal_id=0,color='r',alpha='0.2',vis_thresh=0,dist_thresh=np.inf,recolor_detecting_individuals=False):
		segs=self.pred_segments[focal_id]
		phi=self.phi[focal_id]
		tps=self.tp_obj_cart_pred
		detected=0
		for seg in segs:
			if abs(seg[0]-seg[1])>vis_thresh and self.pos_pred[focal_id,0]<dist_thresh:
				hlp_low=subjpol_to_objcart(self.pos_pred[focal_id,0],seg[0],self.pos[:,focal_id],self.phi[focal_id])
				hlp_high=subjpol_to_objcart(self.pos_pred[focal_id,0],seg[1],self.pos[:,focal_id],self.phi[focal_id])
				p1=line_intersect(hlp_low[0],hlp_low[1],self.pos[0,focal_id],self.pos[1,focal_id],tps[0,0,focal_id],tps[0,1,focal_id],tps[1,0,focal_id],tps[1,1,focal_id])
				p2=line_intersect(hlp_high[0],hlp_high[1],self.pos[0,focal_id],self.pos[1,focal_id],tps[0,0,focal_id],tps[0,1,focal_id],tps[1,0,focal_id],tps[1,1,focal_id])
				visual_area=Polygon([p1,p2,self.pos[:,focal_id]])
				ax.add_artist(visual_area)
				visual_area.set_facecolor(color)
				visual_area.set_alpha(alpha)

				if recolor_detecting_individuals:
					ellipse=Ellipse(self.pos_center[:,focal_id],self.w,1.0,self.phi[focal_id]*180.0/np.pi-90.0)
					ax.add_artist(ellipse)
					ellipse.set_clip_box(ax.bbox)
					ellipse.set_facecolor(color)
					ellipse.set_alpha(1)
					ellipse.set_edgecolor('none')
				detected=1
		return detected
				

	def plot_ellipses(self,fig=None,ax=None,color='seagreen',zorder=100,alpha=0.7,show_index=False,edgecolor='none', cmap=cm.Greys,show_eyes=True, eyecolor='k',eyesize=5,edgewidth=1,z_label='',norm_z=False,show_colorbar=True):
		ellipses=[]
		if fig is None:
			fig=plt.gcf()
		if ax is None:
			ax=plt.gca()
		if type(color)==str or np.shape(color)==(4,) or np.shape(color)==(3,):
			color=[color for i in range(self.n)]
		else:
			cmax=np.amax(color)
			cmin=np.amin(color)
			cmap_z=cmap
			if not norm_z:
				color=cmap((color-cmin)/(cmax-cmin))
				norm_z=cm.colors.Normalize(vmin=cmin,vmax=cmax)
				print('creating norm')
			else:
				color=cmap(norm_z(color))
			
			if show_colorbar:
				ax1 = fig.add_axes([0.2, 0.2, 0.6, 0.03])
				cb_z =colorbar.ColorbarBase(ax1, cmap=cmap_z,norm=norm_z, orientation='horizontal',label=z_label)


		for i in range(self.n):
			ellipses.append(Ellipse(self.pos_center[:,i],self.w,1.0,cast_to_pm_pi(self.phi[i])*180.0/np.pi-90.0))
		for i in range(self.n):
			ax.add_artist(ellipses[i])
			ellipses[i].set_clip_box(ax.bbox)
			ellipses[i].set_facecolor(color[i])
			ellipses[i].set_alpha(alpha)
			ellipses[i].set_edgecolor(edgecolor)
			ellipses[i].set_linewidth(edgewidth)
			ellipses[i].set_zorder(zorder)
			if show_index:
				ax.text(self.pos_center[0,i],self.pos_center[1,i],str(i))
		if show_eyes:
			if eyecolor=='map':
				self.draw_eyes(ax,color=color,size=eyesize)
			else:
				self.draw_eyes(ax,color=eyecolor,size=eyesize)
		ax.set_xlim(np.amin(self.pos_center[0])-1,np.amax(self.pos_center[0])+1)
		ax.set_ylim(np.amin(self.pos_center[1])-1,np.amax(self.pos_center[1])+1)
		ax.set_aspect('equal')

	def color_ellipse(self,i,ax=None,color='r',zorder=10,edgecolor='r',edgewidth=1,alpha=1,color_face=True):
		if ax is None:
			ax=plt.gca()
		a=Ellipse(self.pos_center[:,i],self.w,1.0,self.phi[i]*180.0/np.pi-90.0)
		ax.add_artist(a)
		a.set_zorder(zorder)
		a.set_clip_box(ax.bbox)
		a.set_facecolor(color)
		a.set_alpha(alpha)
		a.set_edgecolor(edgecolor)
		a.set_linewidth(edgewidth)

	def plot_tangent_points(self,ax=None,viewer_id=1,color='orangered',show_points=True,show_lines=False,alpha=.6,size=15):
		if ax is None:
			ax=plt.gca()
		if type(viewer_id)!=list:
			viewer_id=[viewer_id]
		markersize=0
		lw=0
		if show_points:
			markersize=size
		if show_lines:
			lw=50
		tps=self.tp_obj_cart
		for v in viewer_id:
			for i in range(self.n):
				if i!=v:	
						if show_lines:
							ax.plot([tps[0,0,i,v],self.pos[0,v]],[tps[0,1,i,v],self.pos[1,v]],color=color,alpha=alpha)
							ax.plot([tps[1,0,i,v],self.pos[0,v]],[tps[1,1,i,v],self.pos[1,v]],color=color,alpha=alpha)	
						ax.plot(tps[0,0,i,v],tps[0,1,i,v],marker='.',markersize=markersize,color=color,alpha=alpha)
						ax.plot(tps[1,0,i,v],tps[1,1,i,v],marker='.',markersize=markersize,color=color,alpha=alpha)
						
			ax.plot(self.pos[0,v],self.pos[1,v],color=color,markersize=size,marker='.')

	def plot_visual_field(self,ax=None,viewer_id=1,color='0.5',edgewidth=1,alpha=0.4,edgecolor='none',recolor_vis_individuals=False,vis_thresh=0.,dist_thresh=np.inf):
		pos_center=self.pos_center
		pos=self.pos
		phi=self.phi
		segments=self.segments
		tps=self.tp_obj_cart
		md=self.md_center
		if ax is None:
			ax=plt.gca()
		for i in range(self.n):
			if viewer_id!=i:
				for seg in segments[i,viewer_id]:
					if abs(seg[0]-seg[1])>vis_thresh and md[i,viewer_id]<dist_thresh:
							if recolor_vis_individuals:
									ellipse=Ellipse(pos_center[:,i],self.w,1.0,phi[i]*180.0/np.pi-90.0)
									ax.add_artist(ellipse)
									ellipse.set_clip_box(ax.bbox)
									ellipse.set_facecolor(color)
									ellipse.set_alpha(1)
									#ellipse.set_edgecolor(color)
									#ellipse.set_edgecolor('none')
									ellipse.set_linewidth(edgewidth),
									ellipse.set_edgecolor(edgecolor)
							hlp_low=subjpol_to_objcart(md[i,viewer_id],seg[0],pos[:,viewer_id],phi[viewer_id])
							hlp_high=subjpol_to_objcart(md[i,viewer_id],seg[1],pos[:,viewer_id],phi[viewer_id])
							p1=line_intersect(hlp_low[0],hlp_low[1],pos[0,viewer_id],pos[1,viewer_id],tps[0,0,i,viewer_id],tps[0,1,i,viewer_id],tps[1,0,i,viewer_id],tps[1,1,i,viewer_id])
							p2=line_intersect(hlp_high[0],hlp_high[1],pos[0,viewer_id],pos[1,viewer_id],tps[0,0,i,viewer_id],tps[0,1,i,viewer_id],tps[1,0,i,viewer_id],tps[1,1,i,viewer_id])
							visual_area=Polygon([p1,p2,pos[:,viewer_id]])
							ax.add_artist(visual_area)
							visual_area.set_facecolor(color)
							visual_area.set_alpha(alpha)
							#visual_area.set_edgecolor('none')

	def plot_visual_field_without_segments(self,ax=None,viewer_id=1,color='0.5',edgewidth=1,alpha=0.4,edgecolor='none',recolor_vis_individuals=False,vis_thresh=0.,dist_thresh=np.inf):
		pos_center=self.pos_center
		pos=self.pos
		phi=self.phi
		segments=self.visual_field
		tps=self.tp_obj_cart
		md=self.md_center
		colored=[]
		if ax is None:
			ax=plt.gca()
		for k in range(2*(self.n-1)):
			if not np.isnan(segments[0,viewer_id,k]):
				i=int(segments[0,viewer_id,k])
				if self.ang_area[i,viewer_id]>vis_thresh and md[i,viewer_id]<dist_thresh:
					if recolor_vis_individuals and i not in colored:
						colored.append(i)
						ellipse=Ellipse(pos_center[:,i],self.w,1.0,phi[i]*180.0/np.pi-90.0)
						ax.add_artist(ellipse)
						ellipse.set_clip_box(ax.bbox)
						ellipse.set_facecolor(color)
						ellipse.set_alpha(1)
						#ellipse.set_edgecolor(color)
						#ellipse.set_edgecolor('none')
						ellipse.set_linewidth(edgewidth),
						ellipse.set_edgecolor(edgecolor)
					hlp_low=subjpol_to_objcart(md[i,viewer_id],segments[1,viewer_id,k],pos[:,viewer_id],phi[viewer_id])
					hlp_high=subjpol_to_objcart(md[i,viewer_id],segments[2,viewer_id,k],pos[:,viewer_id],phi[viewer_id])
					p1=line_intersect(hlp_low[0],hlp_low[1],pos[0,viewer_id],pos[1,viewer_id],tps[0,0,i,viewer_id],tps[0,1,i,viewer_id],tps[1,0,i,viewer_id],tps[1,1,i,viewer_id])
					p2=line_intersect(hlp_high[0],hlp_high[1],pos[0,viewer_id],pos[1,viewer_id],tps[0,0,i,viewer_id],tps[0,1,i,viewer_id],tps[1,0,i,viewer_id],tps[1,1,i,viewer_id])
					visual_area=Polygon([p1,p2,pos[:,viewer_id]])
					ax.add_artist(visual_area)
					visual_area.set_facecolor(color)
					visual_area.set_alpha(alpha)
					#visual_area.set_edgecolor('none')






				
	def create_network(self,threshold=0.,allinfo=False):
		p=np.copy(self.p)
		if threshold!=0:
			p_notzero=p[np.nonzero(p)]
			thr=np.percentile(p_notzero,threshold)
			low_values_indices = p <= thr
			p[low_values_indices] = 0.0
			print('threshold=%1.6f'%thr)
		network=nx.DiGraph(p)
		if allinfo:
			for i in range(len(p[0])):
				network.nodes()[i]['pos']=self.pos[:,i]
				network.nodes()[i]['phi']=self.phi[i]
		print('created network')
		self.network=network


	def color_ellipses_by(self,fig,ax,z_data,z_label,shadow_size=1.,cmap='Purples',vmin=None,vmax=None):
		ax1 = fig.add_axes([0.2, 0.1, 0.6, 0.03])
		maxx=0.0
		maxy=0.0
		minx=10000
		miny=10000
		if vmin is None:
			vmin=np.amin(z_data)
		if vmax is None:
			vmax=np.amax(z_data)
		cmap_z=cm.get_cmap(cmap)
		norm=cm.colors.Normalize(vmin=vmin,vmax=vmax)
		cb_z =colorbar.ColorbarBase(ax1, cmap=cmap_z,norm=norm, orientation='horizontal',label=z_label,alpha=0.85)
		
		for n in self.network:
			c=Ellipse(self.network.nodes[n]['pos']+np.array([-self.l/2.0*np.cos(self.network.nodes[n]['phi']),-self.l/2.0*np.sin(self.network.nodes[n]['phi'])]),self.w,1.0,self.network.nodes[n]['phi']*180.0/np.pi-90.0,zorder=1000000)

			ax.add_patch(c)#,zorder=zorder)
			c.set_clip_box(ax.bbox)
			c.set_linewidth(2.5)
			#c.set_facecolor(cmap_out((z_data[k]-min_z)/(max_z-min_z)))#z variable 
			c.set_facecolor(cmap_z(norm(z_data[n])))#z variable 
			c.set_alpha(0.65)
			c.set_edgecolor('none')
			if shadow_size>self.w:
				c_shadow=Ellipse(self.network.nodes[n]['pos']+np.array([-self.l/2.0*np.cos(self.network.nodes[n]['phi']),-self.l/2.0*np.sin(self.network.nodes[n]['phi'])]),self.w*shadow_size,1.0*shadow_size,self.network.nodes[n]['phi']*180.0/np.pi-90.0,zorder=999999)
				ax.add_patch(c_shadow)
				c_shadow.set_clip_box(ax.bbox)
				c_shadow.set_linewidth(2.5)
				#c.set_facecolor(cmap_out((z_data[k]-min_z)/(max_z-min_z)))#z variable 
				c_shadow.set_facecolor(cmap_z(norm(z_data[n])))#z variable 
				c_shadow.set_alpha(0.2)
				c_shadow.set_edgecolor('none')
			self.network.nodes[n]['patch']=c
			ax.plot(self.network.nodes[n]['pos'][0],self.network.nodes[n]['pos'][1],marker='.',color='w',markersize=5,markeredgecolor='w')
			if self.network.nodes[n]['pos'][0]>maxx:
				maxx=self.network.nodes[n]['pos'][0]
			if self.network.nodes[n]['pos'][1]>maxy:
				maxy=self.network.nodes[n]['pos'][1]
			if self.network.nodes[n]['pos'][0]<minx:
				minx=self.network.nodes[n]['pos'][0]
			if self.network.nodes[n]['pos'][1]<miny:
				miny=self.network.nodes[n]['pos'][1]


	def draw_eyes(self,ax,color='k',size=20):
		ax.scatter(self.pos[0,:],self.pos[1,:],color=color,s=size,zorder=10000)

	def draw_network(self,fig=None,ax=None,vary_link_width=False, draw_ellipses=True,link_zorder=10,scale_arrow=10,threshold=0.05,link_cmap=cm.Greys,linkalpha=0.8,lw=2,arrowstyle='-|>',upper_limit=0.9,lower_limit=0.1, draw_colorbar=True, fixedcolorrange=None):
		'''
		INPUT:

		network			nx.DiGraph(p)
		threshold	all links with a weight>=threshold are plotted
		'''
		if fig is None:
			fig=plt.gcf()
		if ax is None:
			ax=plt.gca()
		l=self.l
		w=self.w	
		network=self.network
		ds=list(nx.get_edge_attributes(network,'weight').values())
		#mind=np.amin(ds)
		#maxd=np.amax(ds)*upper_limit
		#print(mind,maxd)
		if fixedcolorrange is None:
			mind=np.percentile(ds,lower_limit*100.)
			maxd=np.percentile(ds,upper_limit*100.)
		else:
			mind=fixedcolorrange[0]
			maxd=fixedcolorrange[1]
		cmap_tmp=link_cmap
		#cmap_edge=cmap_tmp
		cmap_edge=truncate_colormap(cmap_tmp,0.2,1.)
		norm_edge=cm.colors.Normalize(vmin=mind,vmax=maxd)
		if draw_colorbar:
			ax2 = fig.add_axes([0.3, 0.1, 0.4, 0.03])	
			cb_edge =colorbar.ColorbarBase(ax2, cmap=cmap_edge, norm=norm_edge,orientation='horizontal',label='Response Rate',drawedges=False)

		for n in network:
			c=Ellipse(network.nodes[n]['pos']+np.array([-l/2.0*np.cos(network.nodes[n]['phi']),-l/2.0*np.sin(network.nodes[n]['phi'])]),w,1.0,network.nodes[n]['phi']*180.0/np.pi-90.0)
			ax.add_patch(c)
			c.set_clip_box(ax.bbox)
			c.set_linewidth(2)
			#c.set_facecolor(cmap_out((z_data[k]-min_z)/(max_z-min_z)))#z variable 
			c.set_facecolor('none') 
			c.set_alpha(1)
			if draw_ellipses:
				c.set_edgecolor('k')
			else:
				c.set_edgecolor('none')
			network.nodes[n]['patch']=c
		seen={}
		for (u,v,d) in network.edges(data=True):
			if vary_link_width:
				lw=6*(d['weight']-mind)/(maxd-mind)+0.5
			if d['weight']>=threshold:
				n1=network.nodes[u]['patch']
				n2=network.nodes[v]['patch']
				rad=0.05
				if (u,v) in seen:
					rad=seen.get((u,v))
					rad=(rad+np.sign(rad)*0.1)*-1
				e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
								arrowstyle=arrowstyle,
								mutation_scale=scale_arrow,
								connectionstyle='arc3,rad=%s'%rad,
								lw=lw,
								alpha=linkalpha,
								color=cmap_edge((d['weight']-mind)/(maxd-mind)),zorder=link_zorder)
				seen[(u,v)]=rad
				ax.add_patch(e)
		#positions=nx.get_node_attributes(network,'pos').values()
			# ax.plot(positions[:,0],positions[:,1],marker='o',markersize=3,markerfacecolor='k',markeredgewidth=0,linewidth=0)
		if draw_colorbar:
			box = ax.get_position()
			ax.set_position([box.x0, box.y0+box.height*0.2, box.width, box.height*0.8])
	
	def draw_binary_network(self,fig=None,ax=None,rad=0.0, draw_ellipses=True,link_zorder=10,show_index=False,scale_arrow=10,linkalpha=0.8,lw=0.8,arrowstyle='-|>',linkcolor='k'):
		'''
		INPUT:

		network			nx.DiGraph(p)
		
		'''
		if not self.network.has_node(0):
			self.create_network(allinfo=True)
		if fig is None:
			fig=plt.gcf()
		if ax is None:
			ax=plt.gca()
		l=self.l
		w=self.w	
		network=self.network
		for n in network:
			if show_index:
				ax.text(network.nodes[n]['pos'][0],network.nodes[n]['pos'][1],str(int(n)))	
			c=Ellipse(network.nodes[n]['pos']+np.array([-l/2.0*np.cos(network.nodes[n]['phi']),-l/2.0*np.sin(network.nodes[n]['phi'])]),w,1.0,network.nodes[n]['phi']*180.0/np.pi-90.0)
			ax.add_patch(c)
			#ax.plot(network.nodes[n]['pos'])
			#c.set_clip_box(ax.bbox)
			#c.set_linewidth(2)
			#c.set_facecolor(cmap_out((z_data[k]-min_z)/(max_z-min_z)))#z variable 
			c.set_facecolor('none') 
			#c.set_alpha(1)
			if draw_ellipses:
				c.set_edgecolor('k')
			else:
				c.set_edgecolor('none')
			network.nodes[n]['patch']=c
		seen={}
		
		for (u,v,d) in network.edges(data=True):
		
			#if d['weight']>=threshold:
			n1=network.nodes[u]['patch']
			n2=network.nodes[v]['patch']
			#rad=0.05
			if (u,v) in seen:
				rad=seen.get((u,v))
				rad=(rad+np.sign(rad)*0.1)*-1
			e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
								arrowstyle=arrowstyle,
								mutation_scale=scale_arrow,
								connectionstyle='arc3,rad=%s'%rad,
								lw=lw,
								alpha=linkalpha,
								color=linkcolor,zorder=link_zorder)
			seen[(u,v)]=rad
			ax.add_patch(e)
		ax.set_xlim(np.amin(self.pos_center[0])-1,np.amax(self.pos_center[0])+1)
		ax.set_ylim(np.amin(self.pos_center[1])-1,np.amax(self.pos_center[1])+1)
		ax.set_aspect('equal')



	def color_boundary(self,ax,color='indianred',threshold=0.683):
		vis_field=np.sum(self.ang_area,axis=0)/(2.*np.pi)
		boundary=np.where(vis_field< threshold)[0]
		if len(boundary)!=0:
			for i in boundary:
				self.color_ellipse(ax,i,color=color)
	def set_adjmat(self,p):
		self.p=p

	def set_ang_area(self,ang_area):
		self.ang_area=ang_area
	def set_pos_orient(self,pos,phi,pos_offset=0.,center=False):
                 l=self.l
            
                 if center:
                        if np.shape(pos)[0]!=2:
                                if np.shape(pos)[1]==2:
                                        pos_center=pos.T
                                        self.pos_center=pos_center
                                else:
                                        print('positions need to be of shape [2,N] or ([N,2]')
                        else:
                                self.pos_center=pos
                        
                        if len(phi)==np.shape(self.pos_center)[1]:
                                self.pos_center-=np.array([pos_offset/2.0*np.cos(phi),pos_offset/2.0*np.sin(phi)])
                                self.pos=self.pos_center-np.array([-l/2.0*np.cos(phi),-l/2.0*np.sin(phi)])
                                self.phi=phi
                        else:
                                 print('Length of orientations array must correspond to number of given positions')
                 else:
                        if np.shape(pos)[0]!=2:
                                if np.shape(pos)[1]==2:
                                        pos=pos.T
                                        self.pos=pos
                                else:
                                        print('positions need to be of shape [2,N] or ([N,2]')
                        else:
                                self.pos=pos
                        
                        if len(phi)==np.shape(pos)[1]:
                                self.pos-=np.array([pos_offset/2.0*np.cos(phi),pos_offset/2.0*np.sin(phi)])
                                self.pos_center=self.pos+np.array([-l/2.0*np.cos(phi),-l/2.0*p.sin(phi)])
                                self.phi=phi
                        else:
                                print('Length of orientations array must correspond to number of given positions')
                        
                 self.reset_calculated_variables()
                
