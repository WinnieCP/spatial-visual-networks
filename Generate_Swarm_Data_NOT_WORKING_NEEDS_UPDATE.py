import numpy as np
import ellipse_swarm as esw
import itertools as it
import multiprocessing as mp
import h5py 
sentinel = None

def Parallel_Simulation(inqueue, output):
    # This function is used for generating swarms in parallel using multiprocessing
    # Here a swarm of the parameters given in the inqueue is generated and the visual fields of all agents are calculated
    # The result is returned the output

    for state,d,n,nphi,npos,w,st,reposition in iter(inqueue.get, sentinel):  
        i=0
        nosuccess=True
        while i<5 and nosuccess:
                # Because the code that shifts the ellipses to remove overlaps sometimes times out and does not return a result
                # I use this quick and dirty approach where I try to generate a swarm and if it does not work out (i.e. if swarm.calc_visfield returns False)
                # I increase the counter i by one. If there was no success after 5 tries, I move on to the next parameter set. This is not ideal, because it
                # may lead to fewer samples being generated than you intended (e.g. when setting stats=50, you may still end up with less than 50 swarms per
                # parameter combination. I have not found a better solution yet. You will see a message printed in the console, see if i==5 below.
                try:
                    swarm=esw.Swarm(setup=state,N=n,dist=d,noise_phi=nphi,noise_pos=npos,w=w,l=0.)





########################## Start replacing with new code below this line ########################################################



                    ok=swarm.calc_visfield(reposition=reposition) 
                    print('calculated visual field')
                    if ok:    
                        print('worked ok')
                        swarm.calc_only_vis_angle()
                        hxhy=np.vstack([np.cos(swarm.phi),np.sin(swarm.phi)])
                        data=[swarm.pos,hxhy,swarm.ang_area,swarm.md_center,swarm.vis_angles]
                        param_groupname='/'+state+'/N%i'%n+'/w%1.2f'%w+'/noisePos%1.3f'%npos+'/noisePhi%1.5f'%nphi+'/dist%1.3f'%d
                        data_name_list=['positions','hxhy','angularArea','metricDistances','angularAreaNoOcclusions']
                        nosuccess=False
                        output.put([param_groupname,data_name_list,data])


########################### Leave as is after this line #########################################################################                 





                    else:
                        i+=1    
                except:
                    i+=1
                if i==5:
                    print('No success for ',d,n,nphi,npos,w,st)
   
def Simple_Simulation(params):
    # This is used for generating swarms without multiprocessing
    # It is largely identical to the parallel version, just the data is
    # handled differently
    # It would be nice to write the parts that are used in both in an extra function
    # but I haven't gotten around to it

    print(params)
    state,d,n,nphi,npos,w,st,reposition =params
    i=0
    nosuccess=True
    while i<5 and nosuccess:
            #try:
                swarm=esw.Swarm(setup=state,N=n,dist=d,noise_phi=nphi,noise_pos=npos,w=w,l=0.)
                ok=swarm.calc_visfield(reposition=reposition) 
                print('calculated visual field')
                if ok:    
                        print('worked ok')
                        swarm.calc_only_vis_angle()
                        hxhy=np.vstack([np.cos(swarm.phi),np.sin(swarm.phi)])
                        data=[swarm.pos,hxhy,swarm.ang_area,swarm.md_center,swarm.vis_angles]
                        param_groupname='/'+state+'/N%i'%n+'/w%1.2f'%w+'/noisePos%1.3f'%npos+'/noisePhi%1.5f'%nphi+'/dist%1.3f'%d
                        data_name_list=['positions','hxhy','angularArea','metricDistances','angularAreaNoOcclusions']
                        nosuccess=False
                        return [param_groupname,data_name_list,data]
                        
            #except:
                i+=1
                if i==5:
                    print('No success for ',d,n,nphi,npos,w,st)


def handle_output(output):
    # Handles the multiprocessing
    # hdf = h5py.File('/mnt/DATA/swarm_data.h5', 'a') <- This was used before, instead of 
    # the 'with h5py...as hdf' line. Change back if you encouter a problem with this. 
    # don't forget to also change the last line to include the hdf.close() again
    with h5py.File('/mnt/DATA/swarm_data.h5', 'a') as hdf:
        while True:
            args = output.get()
            if args:
                param_groupname, datanames, data = args
                for j,name in enumerate(datanames):
                    if param_groupname+'/'+name not in hdf:
                        #print('creating parameter set group '+param_groupname+'/'+name)
                        #print(np.shape(data[j]))
                        hdf.require_dataset(param_groupname+'/'+name,
                                        data=data[j],shape=[1]+list(np.shape(data[j])),chunks=True,dtype='f',maxshape=(None,None,None))
                    else:   
                        #print('appending to set '+param_groupname)
                        hdf[param_groupname+'/'+name].resize((hdf[param_groupname+'/'+name].shape[0]+1),axis=0)
                        hdf[param_groupname+'/'+name][-1]=data[j]
            else:
                break
   # hdf.close() 

    
if __name__=="__main__":
    parallel=True
    reposition=False # whether to shift the ellipses to avoid overlaps, not eliminating overlaps will cause errors in the visual field calculations at high density
    num_processors=5
 
# Set parameters as lists, all combinations are calculated
    states=['grid']
    dist=np.array([5.])#np.array([0.5, 0.67, 0.84, 0.92, 1.0, 1.1, 1.18, 1.25, 1.38, 1.5, 1.75, 10.0, 2.0, 2.5, 20.0, 3.0, 4.0, 5.0, 7.5])#np.array([0.92,1.18,1.38])#np.array([0.5,1.,1.25,1.5,2.,3.,5.,10.,20.])#np.hstack([np.arange(0.4,2,0.2),[3,4,5,10,20]])
    number=np.array([400])
    noisephi=np.logspace(-1, 1.5, 5)[1:4]#[[1,2,4,5]]
    noisepos=[0.5]
    aspect=[0.3]#,0.4,0.6,0.8]#,0.5,0.9]
    stats=np.arange(30) # how many configurations for each set of parameters

# Set up multiprocessing
    processes=len(dist)*len(number)*len(noisephi)*len(noisephi)*len(aspect)*len(stats)
    if processes<num_processors:
        num_processors=processes
    print('number of processes ',num_processors)
    paramlist= it.product(states,dist,number,noisephi,noisepos,aspect,stats,[reposition])
 
# Multiprocessing that writes into one HDF5 file. Because writing into the file can not be done in parallel by many processes, the results need to be queued before
    if parallel:
        output = mp.Queue()
        inqueue = mp.Queue()
        jobs = []
        proc = mp.Process(target=handle_output, args=(output, ))
        proc.start()
        for i in range(num_processors):
            p = mp.Process(target=Parallel_Simulation, args=(inqueue, output))
            jobs.append(p)
            p.start()
        for i in paramlist:
            inqueue.put(i)
        for i in range(num_processors):
            # Send the sentinal to tell Simulation to end
            inqueue.put(sentinel)
        for p in jobs:
            p.join()
        output.put(None)
        proc.join()
# Version without parallel processing and saving each swarm to a txt file. I have only used this for debugging and you would have to change it to save to a SINGLE hdf5, if you want to run code on a single core only (without multiprocessing) and then use the process_swarm_data.py for processing (or simply use the parallel version above, which already generates the right hdf5 file)
    else:
        data=[]
        i=0
        for params in paramlist:
            data.append(Simple_Simulation(params))
            i+=1
            if i==3:
                break
        print(np.shape(data))
        for dat in data:
            for k,da in enumerate(dat[1]):
                print(da)
                print('saving to '+dat[0].replace('/','_')+'_'+da+'.txt')
                np.savetxt(dat[0].replace('/','_')+'_'+da+'.txt',dat[2][k])
