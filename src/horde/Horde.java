package horde;

import java.util.LinkedList;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;

public class Horde {
	
	private final static boolean onlyAvailable = true;
	
	CLPlatform platform;
	CLContext[] contexts;
	CLQueue[] queues;
	CLDevice[] devices;
	GPUHorde[] hordes;
	
	/**
	 * Initialise all the OpenCL contexts and pick the best platform on which to run
	 */
	public void init(){
		// get all platforms containing GPUs
		CLPlatform[] platforms = JavaCL.listGPUPoweredPlatforms();
		LinkedList<CLContext> contextList= new LinkedList<CLContext>();
		int maxGPU=0, maxGPUindex=0;
		
		// create a context for every GPU from the platform containing the most GPUs
		for(int i=0; i< platforms.length; i++){
			devices= platforms[i].listGPUDevices(onlyAvailable);
			if(maxGPU< devices.length){
				maxGPU= devices.length;
				maxGPUindex=i;
				contextList.clear();
				for(int j=0; j< devices.length; j++){
					contextList.add(platforms[i].createContext(null, devices[j]));
				}
			}
		}
		// set and print info
		platform= platforms[maxGPUindex];
		printPlatformInfo();
		
		devices= platform.listGPUDevices(onlyAvailable);
		printDeviceInfo();
		
		// save all the new context in the array contexts
		contexts= new CLContext[contextList.size()];
		contexts= contextList.toArray(contexts);
		
		// create a queue for every context and create GPUHorde for every GPU
		queues= new CLQueue[contexts.length];
		hordes= new GPUHorde[contexts.length];
		for(int i=0; i< contexts.length; i++){
			queues[i]= contexts[i].createDefaultQueue();
			hordes[i]= new GPUHorde(contexts[i], queues[i], devices[i]);
		}
		
		
	}
	
	public void printPlatformInfo(){
		System.out.println("Currently using:");
		System.out.println("Platform:\t" + platform.getName());
		System.out.println("\t\tOpencl " + platform.getVersion());
	}
	
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
	
	public void start(){
		
	}
	
	public void step(){
		
	}
	
	public void end(){
		
	}
	
}
