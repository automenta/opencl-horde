/************************************************************
* Also, for now, this implementation does not make use of   *
* local memory. Future implementation could use the local   *
* memory if enough is available. This could lead to great   *
* speedups                                                  *
************************************************************/

/* Possible optimization in vecotrizing some of the data */

#ifndef ALPHA
#define ALPHA 0.1f
#endif

#ifndef ETA
#define ETA 0.1f
#endif

#ifndef LAMBDA
#define LAMBDA 0.6f
#endif

/*
* 	Returns the index of the next bit set to one starting at a
* specific index (including that index)
*
* Param
*	index :		The index at which to start
*	
*	x :		The integer we want to compute the next leading
*			one.
*
* Return
*	i :		The index of the next bit set to one in x
*/
int nextLeadingOne(uint index, uint x){
	x=x << index;
	int i=clz(x)+index;
	if(i>= sizeof(uint)){
		return -1;
	}else{
		return i;
	}

}

/*
*	Compute delta according to GTD Lambda
*
* Param
* 	theta : 	The weights that approximate Q(s,a)
*
* 	features1 :	The feature vector representing s_t
*
* 	features2 :	The feature vector representing s_(t+1)
*
* 	gamma :		The discount factor
*
* 	reward :	The reward
*
*	index :		The index of that specific demon (might be better to query it)
*
*	dim :		The dimension of the feature vector
*
*	numDemons :	The total number of demons (might be better to query it)
*
*
* Return 
*	delta:	The computed delta
*
*/ 
float2 computeDeltaGTD( __global const float* theta, 
			__global const float* features1,
			__global const float* features2, 
			float gamma, 
			float reward,
			int index,
			int dim,
			int numDemons)
{
	int i;
	int j=0;
	float Q1=0.0f, Q2=0.0f;
	float2 delta;
	for(i=index; i<dim*numDemons; i+= numDemons){
		Q1 += theta[i]*features1[j];
		Q2 += theta[i]*features2[j];
		j++;
	}
	delta.x= reward + gamma*Q2 - Q1;
	delta.y= Q2;
	return delta;
}

/*
*	Update theta according to GTD(lambda)
*
* Param
*	theta :		The weights that approximate Q(s,a)
*
*	w :		The w weights in the GTD(lambda) algorithm
*
*	trace :		The eligibility trace
*
* 	features1 :	The feature vector representing s_t
*
*	alpha :		The learning rate
*
*	gamma :		The discount factor
*
*	delta :		Computed delta according to the GTD(lambda) algorithm
*
*	lambda :	Rate of decay of the elgibility trace
*
*	index :		The index of that specific demon (might be better to query it)
*
*	dim :		The dimension of the feature vector
*
*	numDemons :	The total number of demons (might be better to query it)	
*
*/
void updateThetaGTD(	__global float* theta,
			__global const float* w,
			__global const float* trace,
			__global const float* features1,
			float alpha,
			float gamma,
			float delta,
			float lambda,
			int index,
			int dim,
			int numDemons)
{
	int i;
	int j=0;
	float one_minus_lambda= 1.0f-lambda;
	float Qw=0.0f;
	for(i=index; i<dim*numDemons; i+= numDemons){
		Qw += trace[i]*w[i];
	}

	for(i=index; i<dim*numDemons; i+= numDemons){
		theta[i] = theta[i] + alpha*(delta*trace[i] 
				- gamma*(one_minus_lambda)*Qw*features1[j]);
		j++;
	}
}

/*
*	Update w according to GTD(lambda)
*
* Param
*	w :		The w weights in the GTD(lambda) algorithm
*
*	trace :		The eligibility trace
*
* 	features1 :	The feature vector representing s_t
*
*	alpha :		The learning rate
*
*	delta :		Computed delta according to the GTD(lambda) algorithm
*
*	index :		The index of that specific demon (might be better to query it)
*
*	dim :		The dimension of the feature vector
*
*	numDemons :	The total number of demons (might be better to query it)	
*
*/
void updateWGTD(__global float* w,
		__global const float* trace,
		__global const float* features1,
		float alpha,
		float delta,
		int index,
		int dim,
		int numDemons)
{
	int i;
	int j=0;
	float Qw=0;
	for(i=index; i<dim*numDemons; i+= numDemons){
		Qw += features1[j]*w[i];
		j++;
	}
	j=0;
	for(i=index; i<dim*numDemons; i+= numDemons){
		w[i] += w[i] + alpha*(delta*trace[i] - Qw*features1[j]);
		j++;
	}
}

/*
*	Update the eligibility trace according to GTD(lambda)
*
* Param
*	trace :		The eligibility trace
*
*	features1 :	The feature vector
*
*	rho :		The importance sampling
*
*	gamma :		The discount factor
*
*	lambda :	The trace decay rate
*
*	index :		The index of that specific demon (might be better to query it)
*
*	dim :		The dimension of the feature vector
*
*	numDemons :	The total number of demons (might be better to query it)
*
*/
void updateTraceGTD(	__global float* trace,
			__global const float* features1,
			float rho,
			float gamma,
			float lambda,
			int index,
			int dim,
			int numDemons)
{
	int i;
	int j=0;
	for(i=index; i<dim*numDemons; i+= numDemons){
		trace[i]= rho*(features1[j] + gamma*lambda*trace[i]);
		j++;
	}
}


__kernel void
 updateGTDLambda(__global float* theta, 
		__global float* w,
		__global float* trace, 
		__global const float* features1,
		__global const float* features2, 
		__global const float* rhoArray, 
		__global const float* rewardArray,
		__global const float* gammaArray,
		__global float* prediction,
		int dim)
{
	int index= get_global_id(0);
	int numDemon= get_global_size(0);
	float rho= rhoArray[index];
	float reward= rewardArray[index];
	float gamma= gammaArray[index];

	//Compute the TD error
	float2 delta= computeDeltaGTD(theta, features1, features2, gamma, reward, index, dim, numDemon);

	//update the prediction
	prediction[index]= delta.y;

	//Update the elligibility trace
	updateTraceGTD(trace, features1, rho, gamma, LAMBDA, index, dim, numDemon);

	//Update Theta
	updateThetaGTD(theta, w, trace, features1, ALPHA, gamma, delta.x, LAMBDA, index, dim, numDemon);
	
	//Update w
	updateWGTD(w, trace, features1, ALPHA*ETA, delta.x, index, dim, numDemon);

}

/*
*	Simple initialization method that sets all weights and trace to zero
*
* Param
*	theta :		The weights that approximate Q(s,a)
*
*	w :		The w weights in the GTD(lambda) algorithm
*
*	trace :		The eligibility trace
*
*
*/
__kernel void initialise(__global float* theta, 
		__global float* w,
		__global float* trace, 
		int dim)
{
	int index= get_global_id(0);
	int numDemons= get_global_size(0);
	int i;
	for(i=index; i<dim*numDemons; i+= numDemons){
		theta[i]=0;
		w[i]=0;
		trace[i]=0;
	}
}

/*
*	Fetch the predictions of all demons given a feature vector
*
* Param
*	theta :		The weights that approximate Q(s,a)
*
*	features :	The feature vector on which the predictions are based
*
*	predictions :	The buffer where the predictions are stored
*
*
*/
__kernel void predict(__global const float* theta,
		__global const float* features,
		__global float* predictions,
		int dim)
{
	int index= get_global_id(0);
	int numDemons= get_global_size(0);
	int i;
	int j=0;
	float Q=0.0f;
	for(i=index; i<dim*numDemons; i+= numDemons){
		Q += theta[i]*features[j];
		j++;
	}
	predictions[index]=Q;
}

