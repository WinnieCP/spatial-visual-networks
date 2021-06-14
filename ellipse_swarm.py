import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
from matplotlib.patches import Ellipse,Polygon,FancyArrowPatch
import networkx as nx
import os
import shutil

from shapely import geometry
from shapely.geometry.polygon import LinearRing

class Swarm:

        '''#####################################################################
        
        This class calculates and stores all data on the swarm of ellipses
                pos:                            position of the eye of each ellipse (2xN array)

                pos_center:                     position of the center of each ellipse (2xN array)

                phi:                            orientation of each ellipse (phi=0 towards pos. x-axis)

                w:                              width of the ellipse, float

                l:                              twice the offset of the eye on the ellipse main axis
                                                from the center, float (l=-1 back, l=1 front, l=0 center)

                n:                              number of ellipses (just for convenience), int

                metric_distance_center:         array of shape n*n, euclidean distance of ellipses 
                                                centers to another 

                tangent_pt_subj_pol:            array of shape [2,n,n] containing the 2 tangent 
                                                points (as 2d arrays) for all combinations of 
                                                ellipses. Indices [i,j,k]:
                                                        i: 1st or second tangent point
                                                        j: id of target (tps lie on this ellipse)
                                                        k: id of observer (this ellipses contains the eye)
                                                entries are 2d arrays containing r, theta of tangent
                                                point in polar coordinates with the eye of the ob-
                                                server in the origin and theta=0 on pos x-axis
                                                
                tangent_pt_obj_cart:            array of shape [2,n,n] containing the 2 tangent 
                                                points (as 2d arrays) for all combinations of 
                                                ellipses. Indices [i,j,k]:
                                                        i: 1st or second tangent point
                                                        j: target (tps lie on this ellipse)
                                                        k: observer (this ellipses contains the eye)
                                                entries are 2d arrays containing x,y of tangent
                                                point in the objective cartesian coordinates (same
                                                as used for pos)

                visual_angles:     array (n*n) used to save the visual angle an individual i would have in the 
                                                visual field of individual j (second index) if not obscured
                                                by any others
                
        #####################################################################'''


        def __init__(self,N=40,setup='grid',pos=None,pos_center=None,phi=None,w=0.4,l=0.,dist=2.,noise_pos=0.1,noise_phi=0.9,eliminate_overlaps=True):
                ''' 
                When an instance of swarm is initialized this sets:
                        - pos: the position of the eye,
                        - pos_center: position of the geometric center of each ellipse,
                        - phi: orientations of the ellipses w.r.t. positive x-axis
                        - w: width of the ellipses
                        - l: half of the distance of the eye from the center along the
                                 main axis of the ellipse of length 1
                if positions and orientations are given, these are used to create the 
                swarm and any input in N is ignored.
                If no pos and orient are given, they are generated for N ellipses according the input
                parameters:
                        - setup: string, options are 'grid','milling','hexagonal', set basic geometry of spatial configuration
                        - dist: average distance between two grid-neighbors 
                        - noise_pos: float in [0,0.5], determines the level of positional noise added to the basic geometric setup
                                noise is sampled from uniform distribution [-noise_pos*dist,noise_pos*dist]
                        - noise_phi: determines the width of the VonMises distribution that orientational noise is sampled from. 
                                High noise_phi corresponds to low orientational noise
                '''
                self.w=w
                self.l=l

                if pos is None and pos_center is None and phi is None:
                        #initiate positions and orientations according to N, dist, noise_phi, noise_pos and setup
                        pos,phi=self._generate_initial_spatial_configuration(setup,N,noise_pos,dist,noise_phi,w)
                        self.pos_center=pos
                        self.pos=pos-np.array([-l/2.0*np.cos(phi),-l/2.0*np.sin(phi)])
                        self.phi=phi
                        self._reset_calculated_variables()
                else:
                        # set positions and orientations according to input
                        if pos_center is not None and pos is None:
                            self.set_positions_and_orientations(pos_center,phi,center=True)
                        elif pos_center is None and pos is not None:
                            self.set_positions_and_orientations(pos,phi,center=False)
                        else:
                            print('Either pos or pos_center needs to be set. If you intend to generate positions, do NOT set either pos or pos_center or phi.')
                if eliminate_overlaps and self.n>1:
                    ok_to_continue=self._eliminate_overlaps()
                else:
                    ok_to_continue=True
                if ok_to_continue==False:
                    print('Overlaps could not be removed successfully. Please try again.')
       
        def polarization(self):            
            polarization=np.sqrt(np.sum(np.sin(self.phi))**2+np.sum(np.cos(self.phi)**2))/self.n
            return polarization


        def density(self):
            if np.sum(self.metric_distance_center)==0:
                self._calc_metric_distances()
            third_nearest_neighbor_distance = np.sort(self.metric_distance_center,axis=0)[2]
            return 4./(np.pi*np.mean(third_nearest_neighbor_distance)**2)

        def _calc_visual_fields(self):
                '''Calculates the visual field of all ellipses and returns 1 if successfull, 0 if not
                    The calculated quantities are saved in the corresponding properties of the class instance,
                    e.g. self.angular_area
                    '''
                if np.sum(self.metric_distance_center)==0:
                    self._calc_metric_distances()
                self._calc_tangent_pts()
                self._calc_vis_field_and_ang_area()

        def binary_visual_network(self,threshold=0.,return_networkX=False):
            if np.sum(self.angular_area)==0:
                self._calc_visual_fields()
            adjacency_matrix=np.array(self.angular_area>threshold,dtype=int)
            if return_networkX:
                return [adjacency_matrix,self._create_network_graph(adjacency_matrix)]
            else:
                return adjacency_matrix  

        def binary_metric_network(self,threshold=5.,return_networkX=False):
            if np.sum(self.metric_distance_center)==0:
                self._calc_metric_distances()
            adjacency_matrix=np.array(np.nan_to_num(self.metric_distance_center,nan=np.inf)<threshold,dtype=int)
            if return_networkX:
                return [adjacency_matrix,self._create_network_graph(adjacency_matrix)]
            else:
                return adjacency_matrix  

        def binary_topological_network(self,threshold=5,return_networkX=False):
            if np.sum(self.metric_distance_center)==0:
                self._calc_metric_distances()
            adjacency_matrix=np.array(np.argsort(np.argsort(self.metric_distance_center,axis=0),axis=0)<threshold,dtype=int)
            if return_networkX:
                return [adjacency_matrix,self._create_network_graph(adjacency_matrix)]
            else:
                return adjacency_matrix           


        def set_positions_and_orientations(self,pos,phi,center=False):
                '''
                sets the positions of ellipse centers (center=True) or eyes (center=False) as well as orientations,
                resets any measures previously derived from these quantities
                
                INPUT:
                pos: numpy array of dimension 2xN or Nx2
                phi: numpy array or list of length N
                center: boolean
                '''
                l=self.l
                if np.shape(pos)[0]!=2:
                        if np.shape(pos)[1]==2:
                            pos=pos.T
                        else:
                            print('positions need to be of shape [2,N] or ([N,2]')
                            return
                if center:
                        pos_center=pos
                        self.pos_center=pos_center
                        if np.shape(phi)==(np.shape(self.pos_center)[1],):
                                self.pos=self.pos_center-np.array([-l/2.0*np.cos(phi),-l/2.0*np.sin(phi)])
                                self.phi=phi
                        else:
                                 print('Length of orientations array must correspond to number of given positions')
                else:
                        self.pos=pos 
                        if np.shape(phi)==(np.shape(pos)[1],):
                                self.pos_center=self.pos+np.array([-l/2.0*np.cos(phi),-l/2.0*np.sin(phi)])
                                self.phi=phi
                        else:
                                print('Length of orientations array must correspond to number of given positions')
                        
                self._reset_calculated_variables()
 

        
        def _generate_initial_spatial_configuration(self,state,nn,noise_int,d,kappa,w):
       
                if state=='grid':
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

                elif state=='hexagonal':
                        d_y=d/np.sqrt(2.) 
                        n=int(np.floor(np.sqrt(nn)))
                        xlen=n
                        ylen=n
                        number=n*n
                        grid_x=np.linspace(d,d*xlen,xlen,endpoint=True)
                        grid_y=np.linspace(d_y,d_y*ylen,ylen,endpoint=True)
                        x,y=np.meshgrid(grid_x,grid_y)
                        x[0:-1:2]+=d/2.
                        pos=np.array([x.flatten(),y.flatten()])
                        if n<np.sqrt(nn):
                                for i in range(nn-number):
                                        extra=np.array([d*(xlen+1+np.floor(i/n)),d_y*(i%n+1)]).reshape(2,1)
                                        pos=np.hstack([pos,extra])
                        orientations=np.random.vonmises(0.0,kappa,nn)
                        noise_x=(np.random.random((nn))-np.ones((nn))*0.5)*2.0*noise_int*d
                        noise_y=(np.random.random((nn))-np.ones((nn))*0.5)*2.0*noise_int*d_y
                        pos[0]+=noise_x
                        pos[1]+=noise_y
                        return pos,orientations

                elif state=='milling':
                        lower, upper = _numberofrings(nn)
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
                        orientations=_cast_to_pm_pi(orientations)
                        return np.array([xpos,ypos]),orientations

                else:
                        print("state needs to be either milling or grid or hexagonal")



        def _reset_calculated_variables(self):
                ''' Resets all the variables of swarm that are calculated from the original
                        input of positions, orientations, ellipse width w and eye position l
                '''
                self.n=len(self.phi)
                self.metric_distance_center=np.zeros([self.n,self.n])
                self.tangent_pt_subj_pol=np.zeros([2,self.n,self.n])
                self.tangent_pt_obj_cart=np.zeros([2,self.n,self.n])
                self.angular_area=np.zeros([self.n,self.n])
                self.network=nx.DiGraph()
                self.visual_angles=np.zeros([self.n,self.n])
                self.eyeinside=()       
                self.visual_field=np.zeros([self.n,self.n,(self.n-1)*2])
        
        def _eliminate_overlaps(self):
                overlaps_exist=self._check_for_overlaps()
                if overlaps_exist:
                    print('moving ellipses to get rid of intersections')
                    self._reposition_to_eliminate_overlaps()
                overlaps_removed_successfully = not self._check_for_overlaps()    
                return overlaps_removed_successfully
       
        def _calc_metric_distances(self):
                '''
                calculates the euclidean distance between all 
                the geometric centers of the ellipses, accessible 
                via self.metric_distance_center
                '''
                z_center=np.array([[complex(p[0],p[1]) for p in self.pos_center.T]])
                self.metric_distance_center=abs(z_center.T-z_center)

        def _check_for_overlaps(self):
                overlaps_exist=False
                # if any two ellipses are closer than 1 bodylength from each other
                if np.sum(self.metric_distance_center<1.):
                    potential_overlaps=np.array([np.array([a,b]) for a in range(self.n) 
                            for b in range(a) if self.metric_distance_center[a,b]<1.]).T
                    i=0
                    while i in range(len(potential_overlaps[0])):
                        id_1=potential_overlaps[0,i]
                        id_2=potential_overlaps[1,i]
                        if self._check_ellipse_pair_for_overlap(id_1,id_2):
                            overlaps_exist=True
                            i=np.inf
                        i+=1
                return overlaps_exist
                

        def _check_ellipse_pair_for_overlap(self,id1,id2):
                '''determines if ellipse with id1 and ellipse with id2 are intersecting '''
                phi1=self.phi[id1]
                phi2=self.phi[id2]
                pos1=self.pos_center[:,id1]
                pos2=self.pos_center[:,id2]
                w=self.w 
                pos1_eye=self.pos[:,id1]
                pos2_eye=self.pos[:,id2]

                ellipses = [(pos1[0], pos1[1], 1, w/2.0, phi1), (pos2[0], pos2[1], 1, w/2.0, phi2)]
                ellipse_a, ellipse_b =_ellipse_polyline(ellipses)
                are_intersecting = _intersections(ellipse_a,ellipse_b)
                return are_intersecting 

        def _reposition_to_eliminate_overlaps(self,fileName='random',lamda1=0.05, overdamp=0.5):
            '''This function uses C++-code to shift and turn the ellipses
               such that they don't intersect anymore, positions and
               orientations are exchanged via temporary txt files
                    - lamda1 - coefficient of the repulsion area of the cells (their main body) (0.01 - 0.05) \n");
                    - overdamp - coeffiecient that controls cell inertia (0 -1).'''


            # save the current position data to file to be read by C-code
            if fileName=='random':
                    fileName=str(int(np.random.random()*1000000))
            outpath='./position_orientation_data_tmp/'+fileName
            if not os.path.exists(outpath):
                    os.makedirs(outpath)
            pospath=outpath+'_pos.txt'
            headingpath=outpath+'_phi.txt'
            np.savetxt(pospath,self.pos_center.T,fmt='%1.8f')
            np.savetxt(headingpath,self.phi,fmt='%1.8f')
            resultpath=outpath
            # execute the C-code
            command="LD_LIBRARY_PATH=$HOME/lib ./Palachanis2015_particle_system/build/pS 50000 {} {} {} {} {} {} {} {} 1.06 1. 0.".format(180,
                            self.n,(self.w+0.06)/1.06,lamda1,overdamp,pospath,headingpath,resultpath)
            os.system(command)
            #load corrected positions and orientations from C-code output    
            new_pos=np.loadtxt(resultpath+'/pos_d1.000_w%1.2f_bl1.1.txt'%((self.w+0.06)/1.06))
            hxhy=np.loadtxt(resultpath+'/headings_d1.000_w%1.2f_bl1.1.txt'%((self.w+0.06)/1.06))
            new_phi=np.arctan2(hxhy[:,1],hxhy[:,0])

            #set the new positions and orientations
            self.set_positions_and_orientations(new_pos,new_phi,center=True)
            # remove the tmp files
            shutil.rmtree(resultpath)
            os.remove(pospath)
            os.remove(headingpath)

 
        def _calc_tangent_pts(self,check_intersects=True):
                ''' calculates the tangent points of ellipses 
                    the result can be found in 

                    - self.tangent_pt_subj_pol (in polar coordinates 
                        centered at a certain individual)

                    - self.tangent_pt_obj_cart (in cartesian coordinates,
                        same origin as self.pos)

                '''
                #initialize lists to collect the calculated tangent points (tp)
                tp_subj_pol=[]
                tp_obj_cart=[]

                # rename some variables for convenience
                w=self.w
                phi_m=np.array([self.phi,]*self.n).transpose()
                x=self.pos[0]
                y=self.pos[1]
                x_center=self.pos_center[0]
                y_center=self.pos_center[1]

                # calculate the relative positions of i to j in coordinate
                # system with origin in the eye of j
                rel_x=x_center.reshape(len(x_center),1)-x #entry(ij)=pos(i)-pos(j)
                rel_y=y_center.reshape(len(y_center),1)-y
                theta=np.arctan2(rel_y,rel_x)
                z=np.array([[complex(p[0],p[1]) for p in self.pos.T]])
                z_center=np.array([[complex(p[0],p[1]) for p in self.pos_center.T]])
                r=abs(z_center.T-z)
                #indices ij: abs(z_center(i)-z(j)), j is observer, i target
      
                #to avoid errors in further calc. result for these will be set manually
                np.fill_diagonal(self.metric_distance_center,float('NaN'))
                np.fill_diagonal(r,float("NaN"))
                
                # calculate tangent points' parameter psi in parametric ellipse eq.
                psi=_get_tangent_point_parameter(w,r,theta,phi_m)

                for p in psi:
                     # calculate tangent point from psi in local polar coordinates
                     pt_subj_pol=_ellipse_point_from_parameter(r,theta,phi_m,p,w)
                     z_pt_subj_pol=pt_subj_pol[0]+1j*pt_subj_pol[1]
                     theta_tp=_cast_to_pm_pi(np.arctan2(pt_subj_pol[1],pt_subj_pol[0])-self.phi)
                     r_tp=abs(z_pt_subj_pol)
                     np.fill_diagonal(r_tp,0.0)
                     tp_subj_pol.append(np.array([r_tp,theta_tp]))
                     # transform tangent points to cartesian global coordinates
                     pt_obj_cart=pt_subj_pol+np.array([np.array([self.pos[0],]*self.n),np.array(\
                     [self.pos[1],]*self.n)])
                     np.fill_diagonal(pt_obj_cart[0],0.0)
                     np.fill_diagonal(pt_obj_cart[1],0.0)
                     tp_obj_cart.append(pt_obj_cart)
                self.tangent_pt_subj_pol=np.array(tp_subj_pol)
                self.tangent_pt_obj_cart=np.array(tp_obj_cart)
        
        def _calc_vis_field_and_ang_area(self):

                '''1. Calculates the visual field for each ellipse and saves it to 
                   self.visual_field, an nxnx2(n-1) array, indices ijk as follows:
                   i: 0:id of ellipse visible, 1:lower angular boundary of visible section, 2:upper angular boundary of visible section
                   j: viewer id
                   k: which section of visual field
                   (a np.nan entry means no occlusion of visual field in this area)
                   2. then calculates the angular area of each ellipse in the visual field of all 
                   other ellipses and saves it to self.angular_area, a numpy nxn array 
                   indices ij:
                   i: seen individual (the one who is seen by individual j)
                   j: focal individual (the one whose visual field is given)'''

                # get ray angles for each ellipse
                angles=self.tangent_pt_subj_pol[:,1].flatten(order='f')
                angles=np.sort(angles[~np.isnan(angles)].reshape(2*(self.n-1),self.n,order='f').T)
                assert np.logical_and(angles.all()<=np.pi, angles.all()>=-np.pi), 'angles are not in pm pi interval'
                between_angles=_cast_to_pm_pi(np.diff(angles,append=(2.*np.pi+angles[:,0]).reshape(self.n,1),axis=1)/2.+angles)

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
                inters=_get_ellipse_line_intersection_points(eyes,raypoints,self.w)
                inters=_remove_self_intersections(inters,self.n)
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
                closest_id=_get_closest_id(r,out,self.n)

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
                self.angular_area=angular_area
 
        def plot_ellipses(self,fig=None,ax=None,color='w',zorder=100,alpha=0.7,show_index=False,edgecolor='0.4', cmap=cm.Greys,show_eyes=True, eyecolor='k',eyesize=5,edgewidth=1,z_label='',norm_z=False,show_colorbar=True):
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
                        else:
                                color=cmap(norm_z(color))
                        
                        if show_colorbar:
                                ax1 = fig.add_axes([0.2, 0.2, 0.6, 0.03])
                                cb_z =colorbar.ColorbarBase(ax1, cmap=cmap_z,norm=norm_z, orientation='horizontal',label=z_label)


                for i in range(self.n):
                        ellipses.append(Ellipse(self.pos_center[:,i],self.w,1.0,_cast_to_pm_pi(self.phi[i])*180.0/np.pi-90.0))
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
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', colors='0.5')
                ax.spines['bottom'].set_color('0.5')
                ax.spines['left'].set_color('0.5')

 
        def draw_eyes(self,ax,color='k',size=20):
            ax.scatter(self.pos[0,:],self.pos[1,:],color=color,s=size,zorder=10000)

        def plot_visual_field(self,ax=None,viewer_id=1,color='darkseagreen',edgewidth=1,alpha=0.4,edgecolor='none',recolor_vis_individuals=False,vis_thresh=0.,dist_thresh=np.inf):
                pos_center=self.pos_center
                pos=self.pos
                phi=self.phi
                segments=self.visual_field
                tps=self.tangent_pt_obj_cart
                md=self.metric_distance_center
                colored=[]
                if ax is None:
                        ax=plt.gca()
                for k in range(2*(self.n-1)):
                        if not np.isnan(segments[0,viewer_id,k]):
                                i=int(segments[0,viewer_id,k])
                                if self.angular_area[i,viewer_id]>vis_thresh and md[i,viewer_id]<dist_thresh:
                                        if recolor_vis_individuals and i not in colored:
                                                colored.append(i)
                                                ellipse=Ellipse(pos_center[:,i],self.w,1.0,phi[i]*180.0/np.pi-90.0)
                                                ax.add_artist(ellipse)
                                                ellipse.set_clip_box(ax.bbox)
                                                ellipse.set_facecolor(color)
                                                ellipse.set_alpha(1)
                                                ellipse.set_linewidth(edgewidth),
                                                ellipse.set_edgecolor(edgecolor)
                                        hlp_low=_subjpol_to_objcart(md[i,viewer_id],segments[1,viewer_id,k],pos[:,viewer_id],phi[viewer_id])
                                        hlp_high=_subjpol_to_objcart(md[i,viewer_id],segments[2,viewer_id,k],pos[:,viewer_id],phi[viewer_id])
                                        p1=_line_intersect(hlp_low[0],hlp_low[1],pos[0,viewer_id],pos[1,viewer_id],tps[0,0,i,viewer_id],tps[0,1,i,viewer_id],tps[1,0,i,viewer_id],tps[1,1,i,viewer_id])
                                        p2=_line_intersect(hlp_high[0],hlp_high[1],pos[0,viewer_id],pos[1,viewer_id],tps[0,0,i,viewer_id],tps[0,1,i,viewer_id],tps[1,0,i,viewer_id],tps[1,1,i,viewer_id])
                                        visual_area=Polygon([p1,p2,pos[:,viewer_id]])
                                        ax.add_artist(visual_area)
                                        visual_area.set_facecolor(color)
                                        visual_area.set_alpha(alpha)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', colors='0.5')
                ax.spines['bottom'].set_color('0.5')
                ax.spines['left'].set_color('0.5')

                               
        def _create_network_graph(self,adjacency_matrix,allinfo=True,plotting_threshold=0.):
                network=nx.DiGraph(adjacency_matrix)
                if allinfo:
                        for i in range(len(adjacency_matrix[0])):
                                network.nodes()[i]['pos']=self.pos[:,i]
                                network.nodes()[i]['phi']=self.phi[i]
                return network


        def draw_binary_network(self,network,fig=None,ax=None,rad=0.0,draw_ellipses=True,ellipse_edgecolor='k',ellipse_facecolor='none',link_zorder=10,show_index=False,scale_arrow=10,linkalpha=0.5,lw=0.8,arrowstyle='-|>',linkcolor='0.4'):
                '''
                INPUT:

                network                 nx.DiGraph(p)
                
                '''
                if fig is None:
                        fig=plt.gcf()
                if ax is None:
                        ax=plt.gca()
                l=self.l
                w=self.w        
                for n in network:
                        if show_index:
                                ax.text(network.nodes[n]['pos'][0],network.nodes[n]['pos'][1],str(int(n)))      
                        c=Ellipse(network.nodes[n]['pos']+np.array([-l/2.0*np.cos(network.nodes[n]['phi']),-l/2.0*np.sin(network.nodes[n]['phi'])]),w,1.0,network.nodes[n]['phi']*180.0/np.pi-90.0)
                        ax.add_patch(c) 
                        c.set_facecolor(ellipse_facecolor) 
                        if draw_ellipses:
                                c.set_edgecolor(ellipse_edgecolor)
                        else:
                                c.set_edgecolor('none')
                        network.nodes[n]['patch']=c
                seen={}
                
                for (u,v,d) in network.edges(data=True):
                
                        #if d['weight']>=threshold:
                        n1=network.nodes[u]['patch']
                        n2=network.nodes[v]['patch']
                       
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
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', colors='0.5')
                ax.spines['bottom'].set_color('0.5')
                ax.spines['left'].set_color('0.5')

 
def _get_ellipse_line_intersection_points(eyes,tps,w):
    ''' given two points of the line (eyes and tp) calculates
        the points at which this line intersects with an ellipse
        of length 1 and width w with center at the origin and
        orientation along the positive x-axis, 
        returns points as 2x2 array, 
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
    t=(np.array([(-ee-np.sqrt(determinant))/(2.*dd),
        (-ee+np.sqrt(determinant))/(2.*dd)]))
    mask=np.array(t>0.,dtype=float)
    mask[mask==0.]=np.nan
    x=mask*(x1+(x2-x1)*t)
    y=mask*(y1+(y2-y1)*t)
    return np.array([x,y])
    
def _remove_self_intersections(inters,n):
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

def _get_closest_id(r,out,n):

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


def _get_tangent_point_parameter(w,r,theta,phi,main_axis=0.5):

    '''calculates where the tangent points lie on the ellipse, return the corresponding angles,
    these can be translated in to coordinates via using the function
    ellipse_point_from_parameter()
    '''
    w=w/2.0
    aa=np.sqrt(-2.0*main_axis*main_axis*w*w + (main_axis*main_axis + w*w)*r*r +
(w*w - main_axis*main_axis)*r*r*np.cos(2.0*(theta - phi)))/np.sqrt(2.0)
    bb= w*r*np.cos(theta - phi) - main_axis*w
    psi1=2.0*np.arctan2(aa-main_axis*r*np.sin(theta - phi),bb)
    psi2= -2.0*np.arctan2(aa+main_axis*r*np.sin(theta - phi),bb)
    return [psi1,psi2]

def _numberofrings(nn):

        lower_estimate=nn/np.pi
        upper_estimate=(np.sqrt(4.*np.pi*nn+1)+1)/2.*np.pi
        return int(np.floor(lower_estimate)), int(np.floor(upper_estimate))

def _cast_to_pm_pi(a):
        '''Casts any (radian) angle to the 
            equivalent in the interval (-pi, pi)'''
        b = (a+np.pi)%(2.*np.pi)
        b -= np.pi
        return b

def _ellipse_polyline(ellipses, n=100):
    '''returns a polygon approximation of an ellipse with n points'''
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

def _intersections(a, b):
        ea = LinearRing(a)      
        eb = LinearRing(b)
        mp = ea.intersects(eb)
        return mp

def _line_intersect(x1a,y1a,x1b,y1b,x2a,y2a,x2b,y2b):
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
        

def _smallestSignedAngleBetween(x,y): 
    #returns smallest of the two angles from x to y 
    tau=2*np.pi
    a = (x - y) % tau
    b = (y - x) % tau
    return -a if a < b else b

def _ellipse_point_from_parameter(r,theta,phi,psi,w,l=0.5):
    #calculates cartesian coordinates for a point on an ellipse 
    # with long axis 1, short axis w, ellipse center at r,theta
    # that is given by the ellipse parameter psi
    
    x=r*np.cos(theta) + l*np.cos(phi)*np.cos(psi) + w*l*np.sin(phi)*np.sin(psi)
    y=r*np.sin(theta) + l*np.sin(phi)*np.cos(psi) - w*l*np.cos(phi)*np.sin(psi)
    return [x,y]

def _subjpol_to_objcart(r,theta,pos,phi):
    # takes in a point, r, theta from the polar coordinates 
    # with center at pos and orientation phi
    # returns a point in cartesian coordinates (same 
    # coordinate system that pos is given in)
        rot_mat_back=np.array([[np.cos(-phi),np.sin(-phi)],[-np.sin(-phi),np.cos(-phi)]])
        pt_subj=[r*np.cos(theta),r*np.sin(theta)]
        pt_obj=np.dot(rot_mat_back,pt_subj)+pos
        return pt_obj

              
