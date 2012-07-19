package horde;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


import rlpark.plugin.rltoys.envio.actions.Action;
import rlpark.plugin.rltoys.envio.observations.Observation;
import rlpark.plugin.rltoys.horde.functions.GammaFunction;
import rlpark.plugin.rltoys.horde.functions.HordeUpdatable;
import rlpark.plugin.rltoys.horde.functions.OutcomeFunction;
import rlpark.plugin.rltoys.horde.functions.RewardFunction;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.utils.NotImplemented;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
/**
 * This class is meant to mimic the Horde class from rlpark.
 * It sets up many demons on the GPUs which can be all updated at once. It will
 * also partition the workload on several GPUs if it can.
 * 
 * An instance of this class should only be accessed by one thread.
 * 
 * @author Clement Gehring
 *
 */
public class CLHorde {
	
	private final static boolean onlyAvailable = true;
	/**
	 * OpenCL platform
	 */
	CLPlatform platform;
	
	/**
	 * The OpenCL contexts
	 */
	CLContext[] contexts;
	/**
	 * The OpenCL queues
	 */
	CLQueue[] queues;
	/**
	 * The GPUs
	 */
	CLDevice[] devices;
	/**
	 * The GPUHordes that reside in each GPU.
	 */
	GPUHorde[] hordes;
	
	/**
	 * The global list of all demons
	 */
	List<CLDemon> demons;
	
	/**
	 * The global list of all updatable functions
	 */
	List<HordeUpdatable> functions;
	
	/**
	 * The dimensions of the feature vectors
	 */
	private int nbFeatures;
	
	/**
	 * Executor to launch all GPU operations simultaneously
	 */
	ExecutorService executor;
	/**
	 * The wrappers for the update task
	 */
	GPUHordeUpdater[] updaters;
	/**
	 * The wrappers for the predictions task
	 */
	GPUHordepredictor[] predictors;
	Future<?>[] futures;
	
	/**
	 * This class is used to launch all the GPU updates at once
	 * @author Clement Gehring
	 *
	 */
	protected class GPUHordeUpdater implements Runnable{
		GPUHorde horde;
		RealVector x_t, x_tp1;
		Action a_t;
		
		public void set(GPUHorde horde, RealVector x_t, Action a_t, RealVector x_tp1){
			this.horde= horde;
			this.x_t= x_t;
			this.a_t= a_t;
			this.x_tp1= x_tp1;
		}
		
		@Override
		public void run() {
			horde.update(x_t, a_t, x_tp1);
		}
	}
	
	/**
	 * This class is used to launch all the GPU predictions at once
	 * @author Clement Gehring
	 *
	 */
	protected class GPUHordepredictor implements Callable<float[]>{
		GPUHorde horde;
		RealVector v;
		public void set(GPUHorde horde, RealVector v){
			this.horde= horde;
			this.v= v;
		}
		public void set(GPUHorde horde){
			this.horde= horde;
			this.v= null;
		}
		@Override
		public float[] call() {
			if(v==null){
				return horde.predictions();
			}else{
				return horde.predictions(v);
			}
		}
		
	}
	
	/**
	 * Build and initialise the CLHorde.
	 * 
	 * @param demons			The demons
	 * @param rewardFunctions	All the reward functions that will need updating
	 * @param outcomeFunctions	All the outcome functions that will need updating
	 * @param gammaFunctions	All the gamma functions that will need updating
	 * @param nbFeatures		The number of features
	 */
	public CLHorde(List<CLDemon> demons, List<RewardFunction> rewardFunctions, List<OutcomeFunction> outcomeFunctions,
		      List<GammaFunction> gammaFunctions, int nbFeatures) {
		
		// store the demons
		this.demons= new ArrayList<CLDemon>();
		this.demons.addAll(demons);
		functions= new ArrayList<HordeUpdatable>();
		
		// add all functions to the update list
		addFunctions(rewardFunctions);
		addFunctions(outcomeFunctions);
		addFunctions(gammaFunctions);
		
		this.nbFeatures=nbFeatures;
		
		// initialise the OpenCL context and partition the demons
		init();
		partitionDemons();
		
	}
	
	/**
	 * Partitioning of the demons amongst the different GPU.
	 * Currently, the partitioning is very simple and assumes all GPUs
	 * are equivalent.
	 */
	private void partitionDemons() {
		//TODO implement a smarter way of partitioning demons in case where GPU are different
		
		long[] maxAlloc= new long[devices.length];
		long[] maxMem= new long[devices.length];
		long totalMem=0;
		int nbDemons= demons.size();
		
		// compute the total available memory and store the memory info of each GPU
		for(int i=0; i<devices.length; i++){
			maxAlloc[i]= devices[i].getMaxMemAllocSize();
			maxMem[i]= devices[i].getGlobalMemSize();
			totalMem += maxMem[i];
			// check if max buff alloc is big enough
			if(maxAlloc[i] < GPUHorde.getAllocReq(nbFeatures, nbDemons/devices.length)){
				throw new RuntimeException("To small alloc size. Too many demons, too many features");
			}
		}
		
		// check if enough memory is available
		long memReq= 12*nbFeatures*nbDemons + 8*nbFeatures + 12*nbDemons;
		if(totalMem< memReq){
			throw new RuntimeException("Not enough memory on GPUs. Too many demons, too many features");
		}
		
		// Separate the demons amongst the GPUs and initialise their GPUHorde
		ArrayList<CLDemon>[] demonArrays= new ArrayList[devices.length];
		for(int i=0; i<devices.length; i++){
			demonArrays[i]= new ArrayList<CLDemon>(demons.size()/devices.length);
		}
		
		for(int i=0; i< demons.size(); i++){
			int GPUIndex = getGPUindex(i);
			demonArrays[GPUIndex].add(demons.get(i));
		}
		for(int i=0; i<devices.length; i++){
			hordes[i].initialise(demonArrays[i], nbFeatures);
		}
		
		
	}

	/**
	 * Add a needed function that will require updates
	 * @param fn An updateable functions
	 */
	public void addFunctions(HordeUpdatable fn){
		functions.add(fn);
	}
	/**
	 * Add the needed functions that will require updates
	 * @param fns A collection of updateable functions
	 */
	public void addFunctions(Collection<?> fns){
		for( Object fn: fns){
			functions.add( (HordeUpdatable) fn);
		}
	}
	
	
	
	
	
	
	
	/**
	 * Initialise all the OpenCL contexts and pick the best platform on which to run
	 */
	public void init(){
		// get all platforms containing GPUs
		CLPlatform[] platforms = JavaCL.listGPUPoweredPlatforms();
		LinkedList<CLContext> contextList= new LinkedList<CLContext>();
		int maxGPU=0, maxGPUindex=0;
		
		// check if any platform was found
		if(platforms.length == 0){
			throw new RuntimeException("Not OpenCL platform detected. Maybe your opencl drivers are missing.");
		}
		
		// find the platform that offers the most GPUs
		for(int i=0; i< platforms.length; i++){
			devices= platforms[i].listGPUDevices(onlyAvailable);
			if(maxGPU< devices.length){
				maxGPU= devices.length;
				maxGPUindex=i;
			}
		}
		
		// check if a GPU was found
		if(maxGPU == 0){
			throw new RuntimeException("No available GPU found");
		}
		
		// set and print info
		platform= platforms[maxGPUindex];
		printPlatformInfo();

		devices= platform.listGPUDevices(onlyAvailable);
		printDeviceInfo();
		
		// create a context for every GPU
		for(int j=0; j< devices.length; j++){
			contextList.add(platforms[maxGPUindex].createContext(null, devices[j]));
		}
		
		// save all the new context in the array contexts
		contexts= new CLContext[contextList.size()];
		contexts= contextList.toArray(contexts);
		
		// create a queue for every context and create GPUHorde for every GPU
		queues= new CLQueue[contexts.length];
		hordes= new GPUHorde[contexts.length];
		for(int i=0; i< contexts.length; i++){
			queues[i]= contexts[i].createDefaultOutOfOrderQueue();
			hordes[i]= new GPUHorde(contexts[i], queues[i], devices[i]);
		}
		
		// set up executor, the updater runnables and the predictor callables
		executor= Executors.newFixedThreadPool(devices.length);
		updaters= new GPUHordeUpdater[devices.length];
		predictors= new GPUHordepredictor[devices.length];
		futures= new Future<?>[devices.length];
		for(int i=0; i< devices.length; i++){
			updaters[i]= new GPUHordeUpdater();
			predictors[i]= new GPUHordepredictor();
		}
		
	}
	/**
	 * Print basic info of the current platform.
	 */
	public void printPlatformInfo(){
		System.out.println("Currently using:");
		System.out.println("Platform:\t" + platform.getName());
		System.out.println("\t\tOpencl " + platform.getVersion());
	}
	
	/**
	 * Print basic info of the current GPUs used.
	 */
	public void printDeviceInfo(){
		for(int i=0; i< devices.length; i++){
			System.out.println("Device name: "+ devices[i].getName());
			System.out.println("INFO:\n"+ "Vendor:\t\t\t"+devices[i].getVendor());
			System.out.println("Global Memory:\t\t"+devices[i].getGlobalMemSize()/(1024*1024)+" mb");
			System.out.println("Local Memory:\t\t"+devices[i].getLocalMemSize()/ (1024)+ " kb");
			System.out.println("# compute units:\t"+devices[i].getMaxComputeUnits());
			System.out.println("Freq:\t\t\t"+devices[i].getMaxClockFrequency()+ "MHZ");
			System.out.println("Type:\t\t\t"+devices[i].getType());
			System.out.println("max group size:\t\t"+devices[i].getMaxWorkGroupSize());
			System.out.println("max buffer size:\t"+devices[i].getMaxConstantBufferSize()/(1024) +" kb");
			System.out.println("max alloc size:\t\t"+devices[i].getMaxMemAllocSize()/ (1024*1024) + " mb");
		}
	}
	/**
	 * Update the Horde.
	 * First update all functions used by the demons. Then, make every GPU simultaneously update their demons
	 * @param o_tp1
	 * @param x_t
	 * @param a_t
	 * @param x_tp1
	 */
	public void update(Observation o_tp1, RealVector x_t, Action a_t, RealVector x_tp1) {
		// update all functions
		for (HordeUpdatable function : functions){
			function.update(o_tp1, x_t, a_t, x_tp1);
		}
		
		// start all the GPU updates concurrently
		for(int i=0; i< devices.length; i++){
			updaters[i].set(hordes[i], x_t, a_t, x_tp1);
			futures[i]=executor.submit(updaters[i]);
		}
		
		// wait for all to be finished
		for(Future future: futures){
			try {
				future.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		
	}
	/**
	 * Call to compute all predictions
	 * @param v		The feature vector on which to base the predictions
	 * @return		Returns the predictions
	 */
	public float[] predictions(RealVector v){
		float[] p= new float[demons.size()];
		
		// start computing the prediction on all GPUs
		for(int i=0; i<devices.length; i++){
			predictors[i].set(hordes[i], v);
			futures[i]= executor.submit(predictors[i]);
		}
		
		// wait and consolidate the results
		int i=0;
		for( int j=0; j< devices.length; j++){
			float[] ptmp=null;
			try {
				ptmp = (float[]) futures[i].get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			for(int k=0; k< ptmp.length; k++){
				p[i] = ptmp[k];
				i++;
			}
		}
		return p;
	}
	
	/**
	 * Call to compute all predictions based on the last feature vector used
	 * @return		Returns the predictions
	 */
	public float[] predictions(){
		float[] p= new float[demons.size()];
		
		// start computing the prediction on all GPUs
		for(int i=0; i<devices.length; i++){
			predictors[i].set(hordes[i]);
			futures[i]= executor.submit(predictors[i]);
		}
		
		// wait and consolidate the results
		int i=0;
		for( int j=0; j< devices.length; j++){
			float[] ptmp=null;
			try {
				ptmp = (float[]) futures[i].get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			for(int k=0; k< ptmp.length; k++){
				p[i] = ptmp[k];
				i++;
			}
		}
		return p;
	}
	
	/**
	 * Set parameters in all GPUHorde.
	 * This forces all GPUHorde to recompile. Use sparingly.
	 * @param alpha		The new alpha value.
	 * @param eta		The new eta value.
	 * @param lambda	The new lambda value.
	 */
	public void setParam(float alpha, float eta, float lambda){
		for(GPUHorde horde: hordes){
			horde.setParam(alpha, eta, lambda);
		}
	}
	
	public void setParam(int i, float alpha, float eta, float lambda){
		hordes[i].setParam(alpha, eta, lambda);
	}
	
	public float[] getTheta(int index){
		throw new NotImplemented();
	}
	
	/**
	 * Compute index of the GPU to which to demon[globalIndex] belongs to.
	 * @param globalIndex	The global index of a demon
	 * @return		The GPU index.
	 */
	public int getGPUindex(int globalIndex){
		if(globalIndex>= demons.size() || globalIndex< 0) 
			throw new IndexOutOfBoundsException("index: "+Integer.toString(globalIndex));
		
		int numDemon= demons.size(), numDev= devices.length;
		int bigHalf= (numDemon/numDev +1) * numDemon%numDev;
		int GPUIndex=0;
		
		if(globalIndex < bigHalf){
			GPUIndex= globalIndex/ (numDemon/numDev +1);
		}else{
			globalIndex -= bigHalf;
			GPUIndex= (globalIndex/ (numDemon/numDev)) + numDemon%numDev;
		}
		
		return GPUIndex;
	}
	
	/**
	 * Compute the internal index to access demons inside GPUHorde.
	 * @param globalIndex	The global index of a demon
	 * @return		The internal index used by GPUHorde to access demon[globalIndex]
	 */
	public int getLocalIndex(int globalIndex){
		if(globalIndex>= demons.size() || globalIndex< 0) 
			throw new IndexOutOfBoundsException("index: "+Integer.toString(globalIndex));
		
		int numDemon= demons.size(), numDev= devices.length;
		int bigHalf= (numDemon/numDev +1) * numDemon%numDev;
		int localIndex=0;
		
		if(globalIndex < bigHalf){
			localIndex= globalIndex % (numDemon/numDev +1);
		}else{
			globalIndex -= bigHalf;
			localIndex= globalIndex % (numDemon/numDev);
		}
		
		return localIndex;
	}
	/**
	 * When all is done, call this method to shutdown executor.
	 */
	public void shutdown(){
		executor.shutdown();
	}
	
	/**
	 * Fetch the theta weights (unordered).
	 * This is a blocking call and will not fetch all GPUs simultaneously.
	 * @return	The weights in their unordered form
	 */
	public float[] getTheta(){
		//TODO reorder the theta in a way that makes sense. Should be done in GPUHorde
		float[] theta= new float[nbFeatures*demons.size()];
		int k=0;
		for(int i=0; i<hordes.length; i++){
			float[] thetaGPU= hordes[i].getTheta();
			for(int j=0; j<thetaGPU.length; j++){
				theta[k]= thetaGPU[j];
				k++;
			}
		}
		return theta;
		
	}
	
	/**
	 * Fetch the w weights (unordered).
	 * This is a blocking call and will not fetch all GPUs simultaneously.
	 * @return	The weights in their unordered form
	 */
	public float[] getW(){
		//TODO reorder the w weights in a way that makes sense. Should be done in GPUHorde
		float[] w= new float[nbFeatures*demons.size()];
		int k=0;
		for(int i=0; i<hordes.length; i++){
			float[] wGPU= hordes[i].getW();
			for(int j=0; j<wGPU.length; j++){
				w[k]= wGPU[j];
				k++;
			}
		}
		return w;
		
	}
	
	/**
	 * Fetch the w weights (unordered).
	 * This is a blocking call and will not fetch all GPUs simultaneously.
	 * @return	The weights in their unordered form
	 */
	public float[] getTrace(){
		//TODO reorder the w weights in a way that makes sense. Should be done in GPUHorde
		float[] trace= new float[nbFeatures*demons.size()];
		int k=0;
		for(int i=0; i<hordes.length; i++){
			float[] traceGPU= hordes[i].getTrace();
			for(int j=0; j<traceGPU.length; j++){
				trace[k]= traceGPU[j];
				k++;
			}
		}
		return trace;
		
	}
	
	
}
