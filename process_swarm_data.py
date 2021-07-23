import numpy as np
import ellipse_swarm as esw
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
from functools import partial
import multiprocessing as mp
import h5py
import networkx as nx
from scipy.spatial import ConvexHull
import pickle as pkl

def generate_groupname(params):
    # used to read from the hdf5 file that contains all positions, metric distances, angular areas
	dist,n,noisephi,noisepos,w,trial,thresholds,setup,alpha=params
	paramstring=setup+'/N%i/w%1.2f/noisePos%1.3f/noisePhi%1.5f/dist%1.3f'%(n,w,noisepos,noisephi,dist)
	return paramstring

def generate_paramstring(setup=None,N=None,w=None,noisePos=None,noisePhi=None,dist=None):
    # This is needed to get all the subgroups in the hdf5 file. There likely is a better way, but this works
    paramstring=''
    if setup is not None:
        paramstring+=setup
        if N is not None:
            paramstring+='/N%i'%N
            if w is not None:
                paramstring+='/w%1.2f'%w
                if noisePos is not None:
                    paramstring+='/noisePos%1.3f'%noisePos
                    if noisePhi is not None:
                        paramstring+='/noisePhi%1.5f'%noisePhi
                        if dist is not None:
                            paramstring+='/dist%1.3f'%dist
    return paramstring

def get_parameter_lists_from_h5_file(file):
    # This is needed to get all the subgroups in the hdf5 file. There likely is a better way, but this works
    with h5py.File(file,'r') as f:
        all_entries_level={}
        for i in range(6):
            all_entries_level[i]=[]

        all_entries_level[0]=[k for k in f.keys()]

        j=1
        for setup in all_entries_level[0]:
            hlp=[float(k[1:]) for k in f[generate_paramstring(setup=setup)].keys()]
            all_entries_level[j]+=hlp
            j+=1
            for n in all_entries_level[1]:
                hlp=[float(k[1:]) for k in f[generate_paramstring(setup='grid',N=n)].keys()]
                all_entries_level[j]+=hlp
                j+=1
                for w in hlp:
                    hlp=[float(k[8:]) for k in f[generate_paramstring(setup='grid',N=n,w=w)].keys()]
                    all_entries_level[j]+=hlp
                    j+=1
                    for npos in hlp:
                        hlp=[float(k[8:]) for k in f[generate_paramstring(setup='grid',N=n,w=w,noisePos=npos)].keys()]
                        all_entries_level[j]+=hlp
                        j+=1
                        for nphi in hlp:
                            hlp=[float(k[4:]) for k in f[generate_paramstring(setup='grid',N=n,w=w,noisePos=npos,noisePhi=nphi)].keys()]
                            all_entries_level[j]+=hlp
                        j-=1
                    j-=1
                j-=1
        for j in range(5):
            all_entries_level[j]=np.unique(all_entries_level[j])
        print(all_entries_level)
    return all_entries_level

def calc_network_measures(weighted_adjmat):
	#calculates network measures using networkx functions
	binary_adjmat=np.array(weighted_adjmat>0,dtype=int)
	g=nx.DiGraph(incoming_graph_data=binary_adjmat)
	try:
		avg_shortest_path=nx.average_shortest_path_length(g)
	except:
		avg_shortest_path=np.nan
	try:
		clustering=np.array(list(nx.clustering(g).items()))[:,1]
		avg_clustering=np.mean(clustering)
		std_clustering=np.std(clustering)
	except:
		clustering=np.nan
		avg_clustering=np.nan
		std_clustering=np.nan
	try:
		num_con_components=nx.number_connected_components(g.to_undirected())
	except:
		num_con_components=np.nan
	indegree=np.sum(binary_adjmat,axis=0)
	outdegree=np.sum(binary_adjmat,axis=1)
	avg_indegree=np.mean(indegree)
	std_indegree=np.std(indegree)
	avg_outdegree=np.mean(outdegree)
	std_outdegree=np.std(outdegree)
    
	instrength=np.sum(weighted_adjmat,axis=0)
	outstrength=np.sum(weighted_adjmat,axis=1)
	avg_instrength=np.mean(instrength)
	std_instrength=np.std(instrength)
	avg_outstrength=np.mean(outstrength)
	std_outstrength=np.std(outstrength)
    
    
	values=[avg_shortest_path,avg_clustering,std_clustering,num_con_components,avg_indegree,std_indegree,
		avg_outdegree,std_outdegree,avg_instrength,std_instrength,avg_outstrength,std_outstrength]
	return values

def calc_density(params,f):
    # calculates the average density from the average 3rd nearest neighbor distance
	groupname=generate_groupname(params)
	trial=params[5]
	md=np.array(f[groupname+'/metricDistances'])[trial]
	nnd3=np.sort(md,axis=0)[2]
	avg_3nnd=np.mean(nnd3)
	density=4./(np.pi*avg_3nnd**2)
	return density

def calc_polarization(params,f):
	groupname=generate_groupname(params)
	trial=params[5]
	hxhy=np.array(f[groupname+'/hxhy'])[trial]
	if np.shape(hxhy)[1]!=2:
		hxhy=hxhy.T
	polarization=np.sqrt(np.sum(hxhy[:,0])**2+np.sum(hxhy[:,1])**2)/len(hxhy[:,0])
	return polarization

def calc_binary_adjacency_matrix(params,f):
	dist,n,noisephi,noisepos,w,trial,threshold,setup,alpha=params
	groupname=generate_groupname(params)
	if threshold[0]=='visual':
		adjmat=(np.array(f[groupname+'/angularArea'])[trial])>threshold[1]
	else:
		#print(np.shape(np.nan_to_num(np.array(f[groupname+'/metricDistances']),nan=np.inf)))
		metric_distances=np.nan_to_num(np.array(f[groupname+'/metricDistances']),nan=np.inf)[trial]
		if threshold[0]=='metric':
			adjmat=metric_distances<threshold[1]
		elif threshold[0]=='topological':
			adjmat=np.argsort(np.argsort(metric_distances,axis=0),axis=0)<threshold[1]
		else:	
			adjmat=None
			print('Could not find the specified network type: '+threshold[0])
	return adjmat

def add_link_weights(binary_adjmat,params,f):
    # takes a binary adjacency matrix and returns the weighted version
	dist,n,noisephi,noisepos,w,trial,threshold,setup,alpha=params
	groupname=generate_groupname(params)
	metric_distances=np.nan_to_num(np.array(f[groupname+'/metricDistances']),nan=0)[trial]
	weighted_adjmat=calc_link_weight(metric_distances,alpha)*binary_adjmat
	return weighted_adjmat

def calc_weighted_adjacency_matrix(params,f):
	binary_adjmat=calc_binary_adjacency_matrix(params,f)
	weighted_adjmat=add_link_weights(binary_adjmat,params,f)
	return weighted_adjmat

def calc_link_weight(dist,alpha):
	return 1./(1.+dist**alpha)

def calc_avg_rel_linklength(params,f,adjmat):
	dist,n,noisephi,noisepos,w,trial,threshold,setup,alpha=params
	groupname=generate_groupname(params)
	metric_distances=np.nan_to_num(np.array(f[groupname+'/metricDistances']),nan=0)[trial]
	avg_rel_linklength=np.sum(metric_distances*(adjmat>0))/(np.sum(adjmat>0)*np.amax(metric_distances))
	return avg_rel_linklength

def run_data_processing(params):
	dist,n,noisephi,noisepos,w,trial,threshold,setup,alpha,filepath=params
	f=h5py.File(filepath,'r')
	groupname=generate_groupname(params[:-1])
	if groupname in f and trial<np.shape(f[groupname+'/angularArea'])[0]:
		adjMat=calc_weighted_adjacency_matrix(params[:-1],f)
		polarization=calc_polarization(params[:-1],f)
		density=calc_density(params[:-1],f)
		network_measures=calc_network_measures(adjMat)
		avg_rel_linklength=calc_avg_rel_linklength(params[:-1],f,adjMat)
		all_results=[density]+[polarization]+network_measures+[avg_rel_linklength]+[dist,n,noisephi,noisepos,w,trial,threshold[0],threshold[1],setup,alpha]
		f.close()
		return all_results
	else:
		print('does not exist:',params)
		return []

if __name__=="__main__":
	parallel=True # If you set this to False, then the different parameter 
	#sets are just calculated one after the other in a long loop (see below 
	# if parallel, else)
	no_processors=35
	process_entire_file=True
	data_file='/mnt/DATA/swarm_data_noisepos0.5_noisephi0.1_N225_forAnna.h5'

	if process_entire_file:
		#here we get all the parameter values present in the dataset
		all_entries=get_parameter_lists_from_h5_file(data_file)
		setup=all_entries[0]
		number=all_entries[1]
		aspect=all_entries[2]
		noisepos=all_entries[3]
		noisephi=all_entries[4]
		dist=all_entries[5]

	else:
		#set which parameters you want to process 
		#(it's always all possible combinations of them that will be used)
		# if a combination is not in the data set you will see 'does not exist ...'
		# printed in the terminal
		setup=['grid']
		number=[400]
		aspect=[0.3]
		noisepos=[0.5]
		noisephi=np.logspace(-1, 1.5, 5)[1:4]
		dist=[5.]
        
	visual_thresholds=np.array([0.,0.01,0.1])
	alphas=np.linspace(0,2.,3,endpoint=True)
	trials=np.arange(20)
		
    
# I left this structure for the threshold parameter. It is always a tuple, (), containing the
# type of network (here only 'visual') and the value. I thought we might look at metric
# networks at some point which will be easy in this setup

	thresholds=[('visual',th) for th in visual_thresholds]#+[('metric',th) for th in metric_thresholds]#+[('topological',th) for th in topological_thresholds]
	alphas=np.linspace(0,2,3,endpoint=True)
	paramlist= it.product(dist,number,noisephi,noisepos,aspect,trials,thresholds,setup,alphas,[data_file])	
	if parallel:  
		processes=len(setup)*len(visual_thresholds)*len(dist)*len(number)*len(noisepos)*len(noisephi)*len(aspect)*len(trials)
		if processes<no_processors:
			no_processors=processes
		print('number of processes ',no_processors)
		comp_pool=mp.Pool(no_processors)
		data=comp_pool.map(run_data_processing,paramlist)

	else:
		data=[]
		for params in paramlist:
			data.append(run_data_processing(params))

	columnnames=['density','polarization','avg_shortest_path','avg_clustering','std_clustering',
			'number_connected_components','avg_indegree','std_indegree','avg_outdegree',
			'std_outdegree','avg_instrength','std_instrength','avg_outstrength',
			'std_outstrength','avg_rel_linklength','dist','N','noisephi','noisepos','aspect',
			'trial','networktype','threshold','setup','alpha']
	df=pd.DataFrame.from_records(data,columns=columnnames)
	df=df.dropna()
	filename=data_file.split('/')[-1]
	df.to_hdf('/mnt/DATA/processed_'+filename,filename)