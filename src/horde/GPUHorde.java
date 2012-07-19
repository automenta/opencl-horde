package horde;

import java.io.File;
import java.io.IOException;
import java.nio.ByteOrder;
import java.util.List;

import org.bridj.Pointer;

import rlpark.plugin.rltoys.envio.actions.Action;
import rlpark.plugin.rltoys.math.vector.RealVector;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.util.IOUtils;
/**
 * The horde living on a GPU.
 * An instance of this class should be used by only one thread at a time.
 * @author Clement Gehring
 *
 */
public class GPUHorde {
	/**
	 * Buffer that reside on the GPU
	 */
	CLBuffer<Float> thetaBuf, wBuf, traceBuf, rhoBuf, rewardBuf, gammaBuf, predictionBuf;
	/**
	 * Array containing updatable parameter
	 */
	Pointer<Float> gamma, rho, reward;
	
	/**
	 * Feature buffers that reside on the GPU
	 */
	CLBuffer<Float>[] featuresBuf;
	/**
	 * Array containing the feature vectors
	 */
	Pointer<Float>[] features;
	
	/**
	 * OpenCL event that monitors when demons are done updating
	 */
	CLEvent demonUpdate;
	/**
	 * A kernel (program) that will run on the GPU
	 */
	CLKernel updateHorde, predict;
	/**
	 * The dimensions of the kernel tasks
	 */
	int[] numDemon;
	/**
	 * The work group size;
	 */
	int[] workGroupSize= {64};
	
	/**
	 * The context to be used by the GPUHorde
	 */
	CLContext context;
	/**
	 * The Queue to be used by the GPUHorde
	 */
	CLQueue queue;
	/**
	 * The GPU to be used by the GPUHorde
	 */
	CLDevice device;
	
	/**
	 * All the demons that reside on this GPUHorde's GPU
	 */
	List<CLDemon> demons;
	/**
	 * The dimensions of the feature vectors
	 */
	int nbFeatures;
	
	/**
	 * The source code for the kernels
	 */
	String kernelSource;
	/**
	 * The Uncompiled program containing all kernels
	 */
	CLProgram hordeProgram;
	
	/**
	 * The last feature vector used for the predictions
	 */
	RealVector last;
	
	
	public GPUHorde(CLContext context, CLQueue queue, CLDevice device) {
		this.context=context;
		this.queue=queue;
		this.device=device;
	}

	/**
	 * Set up all the  buffers and initialise them on the GPU.
	 * @param demonList A list with all the demons that need to run on that GPU
	 * @param nbFeatures The number of features to handle
	 */
	public void initialise(List<CLDemon> demonList, int nbFeatures) {
		demons= demonList;
		this.nbFeatures= nbFeatures;
		
		features= new Pointer[2];
		featuresBuf= new CLBuffer[2];
		
		ByteOrder order= context.getByteOrder();
		
		// allocate all arrays
		gamma= Pointer.allocateFloats(demons.size()).order(order);
		reward= Pointer.allocateFloats(demons.size()).order(order);
		rho= Pointer.allocateFloats(demons.size()).order(order);
		
		features[0]= Pointer.allocateFloats(nbFeatures).order(order);
		features[1]= Pointer.allocateFloats(nbFeatures).order(order);
		
		// create all buffers to be used on the GPU
		thetaBuf= context.createFloatBuffer(Usage.InputOutput, nbFeatures*demons.size());
		wBuf= context.createFloatBuffer(Usage.InputOutput, nbFeatures*demons.size());
		traceBuf= context.createFloatBuffer(Usage.InputOutput, nbFeatures*demons.size());
		
		gammaBuf= context.createFloatBuffer(Usage.Input, demons.size());
		rhoBuf= context.createFloatBuffer(Usage.Input, demons.size());
		rewardBuf= context.createFloatBuffer(Usage.Input, demons.size());
		predictionBuf= context.createFloatBuffer(Usage.Output, demons.size());
		
		featuresBuf[0]= context.createFloatBuffer(Usage.Input, nbFeatures);
		featuresBuf[1]= context.createFloatBuffer(Usage.Input, nbFeatures);
		
		// load up the kernel source file
		try {
			kernelSource= IOUtils.readText(new File("../horde.cl"));
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		// create the program from source
		hordeProgram= context.createProgram(kernelSource);
		
		
		// create all the kernels and set the arguments
		updateHorde = hordeProgram.createKernel("updateGTDLambda");
		updateHorde.setArgs(thetaBuf, wBuf, traceBuf, featuresBuf[0], featuresBuf[1], rhoBuf, rewardBuf, gammaBuf, predictionBuf, nbFeatures);
		
		predict = hordeProgram.createKernel("predict");
		predict.setArgs(thetaBuf, featuresBuf[0], predictionBuf, nbFeatures);
		
		// initialise all weights using the kernel initialise
		CLKernel initKernel= hordeProgram.createKernel("initialise");
		initKernel.setArgs(thetaBuf, wBuf, traceBuf, nbFeatures);
		
		CLEvent initEvent= initKernel.enqueueNDRange(queue, null, new int[] {demons.size()}, new int[] {64});
		initEvent.waitFor();
		
		// link all demons to their reward, rho and gamma arrays
		for(int i=0; i< demons.size(); i++){
			demons.get(i).initialize(i, reward, rho, gamma);
		}
		
		// set the dimensions of the task
		numDemon= new int[] {demons.size()};
	}

	public void update(RealVector x_t, Action a_t, RealVector x_tp1) {
		
		// update the rewards on the GPU
		for( CLDemon demon: demons){
			demon.updateReward();
		}
		CLEvent rewardWrite= rewardBuf.write(queue, reward, false, demonUpdate);
		
		// update the gammas on the GPU
		for( CLDemon demon: demons){
			demon.updateGamma();
		}
		CLEvent gammaWrite= gammaBuf.write(queue, gamma, false, demonUpdate);
		
		// update the rhos on the GPU
		for( CLDemon demon: demons){
			demon.updateRho(x_t, a_t);
		}
		CLEvent rhoWrite= rhoBuf.write(queue, rho, false, demonUpdate);
		
		// update the feature vectors on the GPU
		double[] d1=x_t.accessData();
		double[] d2=x_tp1.accessData();
		float[] f1= new float[d1.length];
		float[] f2= new float[d2.length];
		for(int i=0; i< d1.length; i++){
			f1[i]= (float) d1[i];
			f2[i]= (float) d2[i];
		}
		
		features[0].setFloats(f1);
		CLEvent feature1Write= featuresBuf[0].write(queue, features[0], false, demonUpdate);
		
		features[1].setFloats(f2);
		CLEvent feature2Write= featuresBuf[0].write(queue, features[0], false, demonUpdate);
		
		//checkForNaN(); //BUG HUNT
		
		// Once all memory transfers are done, run the kernel that will update the weights on the GPU
		CLEvent lastUpdate= demonUpdate;
		last= x_t;
		demonUpdate = updateHorde.enqueueNDRange(queue, numDemon, workGroupSize, rewardWrite, gammaWrite, rhoWrite, feature1Write, feature2Write);
		if(lastUpdate != null)
			lastUpdate.waitFor();
		
	}
	
	/**
	 * Generate the predictions for the given feature vector.
	 * This method will not recompute the predictions if the feature vector did not change
	 * from the last time it was computed.
	 * @param v		The feature vector
	 * @return		An array containing all the predictions. Demon[i] will store in prediction[i].
	 */
	public float[] predictions(RealVector v){
		Pointer<Float> predictions;
		//check if the predictions need to be recomputed
		if(v.equals(last)){
			//if no, then just upload the predictions
			predictions= predictionBuf.read(queue, demonUpdate);
		}else{
			//if yes, send the new feature vector and start the kernel
			double[] d1=v.accessData();
			float[] f1= new float[d1.length];
			for(int i=0; i< d1.length; i++){
				f1[i]= (float) d1[i];
			}
			features[0].setFloats(f1);
			CLEvent feature1Write= featuresBuf[0].write(queue, features[0], false, demonUpdate);
			
			CLEvent predictEvent= predict.enqueueNDRange(queue, numDemon, workGroupSize, feature1Write);
			predictions= predictionBuf.read(queue, predictEvent);
			last=v;
		}
		return predictions.getFloats();
	}
	/**
	 * Fetch the prediction from the last seen feature Vector
	 * @return		An array with the predictions of the last seen feature Vector. Demon[i] will store in prediction[i].
	 */
	public float[] predictions(){
		return predictions(last);
	}

	/**
	 * Set the parameters alpha, eta and lambda.
	 * This is a costly procedure as it needs to recompile the kernel. Use sparingly.
	 * @param alpha		The new alpha.
	 * @param eta		The new eta.
	 * @param lambda	The new lambda.
	 */
	public void setParam(float alpha, float eta, float lambda) {
		if(kernelSource == null){
			throw new RuntimeException("GPUHorde needs to be initialised to change parameters, sorry...");
		}

		// create the program from source
		hordeProgram= context.createProgram(kernelSource);
		
		// define all the required parameters
		hordeProgram.defineMacro("ALPHA", Float.toString(alpha)+"f");
		hordeProgram.defineMacro("ETA", Float.toString(eta)+"f");
		hordeProgram.defineMacro("LAMBDA", Float.toString(lambda)+"f");
		
		// create all the kernels and set the arguments
		updateHorde = hordeProgram.createKernel("updateGTDLambda");
		updateHorde.setArgs(thetaBuf, wBuf, traceBuf, featuresBuf[0], featuresBuf[1], rhoBuf, rewardBuf, gammaBuf, predictionBuf, nbFeatures);
		
		predict = hordeProgram.createKernel("predict");
		predict.setArgs(thetaBuf, featuresBuf[0], predictionBuf, nbFeatures);
		
	}
	
	public float[] getTheta(){
		Pointer<Float> theta= thetaBuf.read(queue, demonUpdate);
		return theta.getFloats();
	}
	
	public float[] getW(){
		Pointer<Float> w= wBuf.read(queue, demonUpdate);
		return w.getFloats();
	}
	
	public float[] getTrace(){
		Pointer<Float> trace= traceBuf.read(queue, demonUpdate);
		return trace.getFloats();
	}
	

	public static long getAllocReq(int nbFeatures, int nbDemons) {
		return 4l*nbDemons*nbFeatures;
	}
	
	public void checkForNaN(){
		float[] f= reward.getFloats();
		for(int i=0; i<f.length; i++){
			if(Float.isNaN(f[i])){
				throw new RuntimeException("reward has NaN");
			}
		}
		
		f=gamma.getFloats();
		for(int i=0; i<f.length; i++){
			if(Float.isNaN(f[i])){
				throw new RuntimeException("gamma has NaN");
			}
		}
		
		f=rho.getFloats();
		for(int i=0; i<f.length; i++){
			if(Float.isNaN(f[i])){
				throw new RuntimeException("rho has NaN");
			}
		}
		
		f=features[0].getFloats();
		for(int i=0; i<f.length; i++){
			if(Float.isNaN(f[i])){
				throw new RuntimeException("feature1 has NaN");
			}
		}
		
		f=features[1].getFloats();
		for(int i=0; i<f.length; i++){
			if(Float.isNaN(f[i])){
				throw new RuntimeException("feature2 has NaN");
			}
		}
		
		f= featuresBuf[0].read(queue, demonUpdate).getFloats();
		for(int i=0; i<f.length; i++){
			if(Float.isNaN(f[i])){
				throw new RuntimeException("featureBuf1 has NaN");
			}
		}
		
		f= featuresBuf[1].read(queue, demonUpdate).getFloats();
		for(int i=0; i<f.length; i++){
			if(Float.isNaN(f[i])){
				throw new RuntimeException("featureBuf2 has NaN");
			}
		}
	}
}
