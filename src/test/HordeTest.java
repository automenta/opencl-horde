package test;

import java.util.ArrayList;
import java.util.Random;

import rlpark.plugin.rltoys.envio.actions.Action;
import rlpark.plugin.rltoys.envio.observations.Observation;
import rlpark.plugin.rltoys.envio.policy.Policy;
import rlpark.plugin.rltoys.horde.functions.ConstantGamma;
import rlpark.plugin.rltoys.horde.functions.ConstantOutcomeFunction;
import rlpark.plugin.rltoys.horde.functions.GammaFunction;
import rlpark.plugin.rltoys.horde.functions.HordeUpdatable;
import rlpark.plugin.rltoys.horde.functions.OutcomeFunction;
import rlpark.plugin.rltoys.horde.functions.RewardFunction;
import rlpark.plugin.rltoys.math.averages.MovingAverage;
import rlpark.plugin.rltoys.math.vector.MutableVector;
import rlpark.plugin.rltoys.math.vector.RealVector;
import zephyr.plugin.core.api.synchronization.Chrono;

import horde.CLDemon;
import horde.CLHorde;

public class HordeTest {

	CLHorde horde, CPUHorde;
	ArrayList<CLDemon> demons;
	ArrayList<RewardFunction> rfns;
	ArrayList<OutcomeFunction> ofns;
	ArrayList<GammaFunction> gfns;
	Random random= new Random();
	
	
	int nbDemons= (int) (4*128*20);
	int nbFeatures= 120;
	
	public class MyrewardFn implements RewardFunction, HordeUpdatable{
		
		private static final long serialVersionUID = -6219067330852249271L;

		public MyrewardFn(){
			super();
		}
		
		@Override
		public double reward() {
			return 0.1;
		}

		@Override
		public void update(Observation o_tp1, RealVector x_t, Action a_t,
				RealVector x_tp1) {
		}
		
	}
	
	public class MyRealVector implements RealVector{

		double[] data;
		
		public MyRealVector(double[] d){
			data= d.clone();
		}
		
		@Override
		public int getDimension() {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public double getEntry(int i) {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public double dotProduct(RealVector other) {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public MutableVector mapMultiply(double d) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public MutableVector subtract(RealVector other) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public MutableVector add(RealVector other) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public MutableVector ebeMultiply(RealVector v) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public MutableVector newInstance(int size) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public MutableVector copyAsMutable() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public RealVector copy() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public double[] accessData() {
			return data;
		}
	}
	
	public class MyOutcome implements OutcomeFunction, HordeUpdatable{

		@Override
		public void update(Observation o_tp1, RealVector x_t, Action a_t,
				RealVector x_tp1) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public double outcome() {
			// TODO Auto-generated method stub
			return 0;
		}

	}
	
	public class MyAction implements Action{

		
	}

	public class RandomPolicy implements Policy{

		int numOfActions;
		
		public RandomPolicy(int nActions){
			numOfActions=nActions;
		}
		@Override
		public double pi(RealVector s, Action a) {
			return 1.0d/numOfActions;
		}

		@Override
		public Action decide(RealVector s) {
			// TODO Auto-generated method stub
			return null;
		}

	}
	
	public void init(){
		rfns= new ArrayList<RewardFunction>(1);
		gfns= new ArrayList<GammaFunction>(1);
		ofns= new ArrayList<OutcomeFunction>(1);
		
		MyrewardFn rewardfn= new MyrewardFn();
		ConstantGamma gammafn= new ConstantGamma(0.1);
		ConstantOutcomeFunction outcomefn= new ConstantOutcomeFunction(0);
		rfns.add(rewardfn);
		
		Policy randomPolicy= new RandomPolicy(2);
		
		demons= new ArrayList<CLDemon>(nbDemons);
		
		
		for(int i=0; i<nbDemons; i++){
			demons.add(new CLDemon(randomPolicy, randomPolicy, rewardfn, gammafn, outcomefn));
		}
		
		try{
			horde= new CLHorde(demons, rfns, ofns, gfns, nbFeatures);
		}catch( RuntimeException r){
			r.printStackTrace();
			System.out.println("Ignoring GPU, still running...\n");
			horde=null;
		}
		
		try{
			CPUHorde= new CLHorde(demons, rfns, ofns, gfns, nbFeatures,true);
		}catch( RuntimeException r){
			r.printStackTrace();
			System.out.println("Ignoring CPU, still running...\n");
			CPUHorde=null;
		}
		
	
	}
	
	public RealVector nextRandomVec(){
		double[] v= new double[nbFeatures];
		v[random.nextInt(nbFeatures)]= 1;
		return new MyRealVector(v);
	}
	
	public void run(){
		init();
		
		System.out.println("nbDemons: "+ nbDemons);
		System.out.println("nbFeatures: "+ nbFeatures);
		
		Chrono chrono= new Chrono();
		MovingAverage average= new MovingAverage(1000);
		
		RealVector x_t=null, x_tp1= null;
		Action a_t=new MyAction();

		
		chrono.start();
		if(horde != null){
			for(int i=0; i<1000; i++){
				x_tp1= nextRandomVec();
				
				long tstart= chrono.getCurrentMillis();
				horde.update(null, x_t, a_t, x_tp1);
				horde.predictions();
				average.update(chrono.getCurrentMillis()- tstart);
				
				if(i%100==0){
					System.out.println("GPUHorde avg= "+ average.average()+" ms");
				}
				x_t= x_tp1;
			}
			horde.shutdown();
		}
		
		if(CPUHorde != null){
			x_t=null;
			for(int i=0; i<1000; i++){
				x_tp1= nextRandomVec();
				
				long tstart= chrono.getCurrentMillis();
				CPUHorde.update(null, x_t, a_t, x_tp1);
				CPUHorde.predictions();
				average.update(chrono.getCurrentMillis()- tstart);
				
				if(i%100==0){
					System.out.println("CPUHorde avg= "+ average.average()+" ms");
				}
				x_t= x_tp1;
			}
			CPUHorde.shutdown();
		}
		
		
	}
	
	public static void main(String[] args){
		(new HordeTest()).run();
	}
}
