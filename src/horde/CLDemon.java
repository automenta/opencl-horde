package horde;

import org.bridj.Pointer;

import rlpark.plugin.rltoys.algorithms.LinearLearner;
import rlpark.plugin.rltoys.envio.actions.Action;
import rlpark.plugin.rltoys.envio.policy.Policy;
import rlpark.plugin.rltoys.horde.demons.Demon;
import rlpark.plugin.rltoys.horde.demons.PredictionOffPolicyDemon;
import rlpark.plugin.rltoys.horde.functions.GammaFunction;
import rlpark.plugin.rltoys.horde.functions.OutcomeFunction;
import rlpark.plugin.rltoys.horde.functions.RewardFunction;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.utils.NotImplemented;

public class CLDemon implements Demon{

	private static final long serialVersionUID = 1684102946950766983L;
	transient int id;
	transient Pointer<Float> rewards;
	transient Pointer<Float> rhos;
	transient Pointer<Float> gammas;
	
	RewardFunction rewardfn;
	GammaFunction gammafn;
	OutcomeFunction outcomefn;
	
	Policy targetPolicy;
	Policy behaviourPolicy;
	
	float[] theta;
	float[] w;
	float[] trace;
	
	public CLDemon(Policy target, Policy behaviour, RewardFunction rewardFunction,
		      GammaFunction gammaFunction, OutcomeFunction outcomeFunction){
		this.targetPolicy=target;
		this.behaviourPolicy= behaviour;
		this.rewardfn= rewardFunction;
		this.gammafn= gammaFunction;
		this.outcomefn= outcomeFunction;
	}

	/**
	 * Updates the arrays corresponding to that demon.
	 * This method does not update the actual values on the GPU. That behaviour is handled in bulk by the 
	 * GPUHorde in charge of this CLDemon.
	 */
	@Override
	public void update(RealVector x_t, Action a_t, RealVector x_tp1) {
		float gamma= (float) gammafn.gamma();
		float reward= (float) (rewardfn.reward() + (1-gamma)*outcomefn.outcome());
		float rho= (float) (targetPolicy.pi(x_t, a_t) / behaviourPolicy.pi(x_t, a_t));
		
		rewards.set(id, reward);
		rhos.set(id, rho);
		gammas.set(id, gamma);
		
	}
	/**
	 * Updates the reward array corresponding to that demon.
	 * This method does not update the actual values on the GPU. That behaviour is handled in bulk by the 
	 * GPUHorde in charge of this CLDemon.
	 */
	public void updateReward(){
		float reward= (float) (rewardfn.reward() + (1-gammafn.gamma())*outcomefn.outcome());
		rewards.set(id, reward);
	}
	
	/**
	 * Updates the gamma array corresponding to that demon.
	 * This method does not update the actual values on the GPU. That behaviour is handled in bulk by the 
	 * GPUHorde in charge of this CLDemon.
	 */
	public void updateGamma(){
		float gamma= (float) gammafn.gamma();
		gammas.set(id, gamma);
	}
	
	/**
	 * Updates the rho array corresponding to that demon.
	 * This method does not update the actual values on the GPU. That behaviour is handled in bulk by the 
	 * GPUHorde in charge of this CLDemon.
	 */
	public void updateRho(RealVector x_t, Action a_t){
		float rho= (float) (targetPolicy.pi(x_t, a_t) / behaviourPolicy.pi(x_t, a_t));
		rhos.set(id, rho);
	}
	
	/**
	 * initialize the CLDemon.
	 * This links the CLDemon to its hosting GPUHorde
	 * @param id The demon id within the GPU
	 * @param rewards The reward array to use
	 * @param rhos The rho array to use
	 * @param gammas The gamma array to use
	 */
	public void initialize(int id, Pointer<Float> rewards, 
			Pointer<Float> rhos, Pointer<Float> gammas){
		
		this.id=id;
		this.rewards=rewards;
		this.rhos=rhos;
		this.gammas=gammas;
	}
	
	/**
	 * store the various weights in the CLDemon.
	 * These values should reflect the demons own values. Some of these array can be null if you desire to not save any.
	 * @param theta theta weights
	 * @param w	w weights
	 * @param trace	trace weights
	 */
	public void setWeights(float[] theta, float[] w, float[] trace){
		this.theta= theta;
		this.w=w;
		this.trace=trace;
	}
	
	/**
	 * Get the stored theta value
	 * @return the theta weights
	 */
	public float[] getTheta(){
		return theta;
	}
	
	/**
	 * Get the stored w value
	 * @return the w weights
	 */
	public float[] getW(){
		return w;
	}
	
	/**
	 * Get the stored trace value
	 * @return the trace weights
	 */
	public float[] getTrace(){
		return trace;
	}

	@Override
	public LinearLearner learner() {
		throw new NotImplemented();
	}
	
	/**
	 * convert a standard off policy demon into the write CLDemon.
	 * Creates a similar CLDemon keeping the same policies and functions and transferring it weights
	 * @param demon The demon that we wish to convert
	 * @return The new CLDemon equivalent
	 */
	public static CLDemon convert(PredictionOffPolicyDemon demon){
		//TODO get the right methods in some of the default demons to automate conversion
		
//		RewardFunction rfn= demon.rewardFunction();
//		OutcomeFunction ofn= demon.outcomeFunction();
//		GammaFunction gfn= demon.gammaFunction();
//		
//		Policy b= demon.behaviourPolicy();
//		Policy t= demon.targetPolicy();
//		
//		float[] theta= demon.learner().weights()b.toFloats();
//		
//		CLDemon newDemon= new CLDemon(t, b, rfn, gfn, ofn);
//		newDemon.setWeights(theta.clone(), null, null);		
//		return newDemon;
		throw new NotImplemented();
	}

}
