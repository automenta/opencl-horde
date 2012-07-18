package horde;

import org.bridj.Pointer;

import rlpark.plugin.rltoys.algorithms.LinearLearner;
import rlpark.plugin.rltoys.envio.actions.Action;
import rlpark.plugin.rltoys.envio.policy.Policy;
import rlpark.plugin.rltoys.horde.demons.Demon;
import rlpark.plugin.rltoys.horde.functions.GammaFunction;
import rlpark.plugin.rltoys.horde.functions.OutcomeFunction;
import rlpark.plugin.rltoys.horde.functions.RewardFunction;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.utils.NotImplemented;

public class CLDemon implements Demon{
	
	private static final long serialVersionUID = 1684102946950766983L;
	int id;
	Pointer<Float> rewards;
	Pointer<Float> rhos;
	Pointer<Float> gammas;
	
	RewardFunction rewardfn;
	GammaFunction gammafn;
	OutcomeFunction outcomefn;
	
	Policy targetPolicy;
	Policy behaviourPolicy;
	
	public CLDemon(Policy target, Policy behaviour, RewardFunction rewardFunction,
		      GammaFunction gammaFunction, OutcomeFunction outcomeFunction){
		this.targetPolicy=target;
		this.behaviourPolicy= behaviour;
		this.rewardfn= rewardFunction;
		this.gammafn= gammaFunction;
		this.outcomefn= outcomeFunction;
	}

	@Override
	public void update(RealVector x_t, Action a_t, RealVector x_tp1) {
		float gamma= (float) gammafn.gamma();
		float reward= (float) (rewardfn.reward() + (1-gamma)*outcomefn.outcome());
		float rho= (float) (targetPolicy.pi(x_t, a_t) / behaviourPolicy.pi(x_t, a_t));
		
		rewards.set(id, reward);
		rhos.set(id, rho);
		gammas.set(id, gamma);
		
	}
	
	public void updateReward(){
		float reward= (float) (rewardfn.reward() + (1-gammafn.gamma())*outcomefn.outcome());
		rewards.set(id, reward);
	}
	public void updateGamma(){
		float gamma= (float) gammafn.gamma();
		gammas.set(id, gamma);
	}
	public void updateRho(RealVector x_t, Action a_t){
		float rho= (float) (targetPolicy.pi(x_t, a_t) / behaviourPolicy.pi(x_t, a_t));
		rhos.set(id, rho);
	}
	
	public void initialize(int id, Pointer<Float> rewards, 
			Pointer<Float> rhos, Pointer<Float> gammas){
		
		this.id=id;
		this.rewards=rewards;
		this.rhos=rhos;
		this.gammas=gammas;
	}

	@Override
	public LinearLearner learner() {
		throw new NotImplemented();
	}

}
