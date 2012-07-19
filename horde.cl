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

#ifndef VECTOR
#define VECTOR float4
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
float2 computeDeltaGTD( __global float* theta, 
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
			__global float* w,
			__global float* trace,
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
		__global float* trace,
		__global const float* features1,
		float alpha,
		float delta,
		int index,
		int dim,
		int numDemons)
{
	int i;
	int j=0;
	float Qw=0.0;
	for(i=index; i<dim*numDemons; i+= numDemons){
		Qw += features1[j]*w[i];
		j++;
	}
	j=0;
	for(i=index; i<dim*numDemons; i+= numDemons){
		w[i] = w[i] + alpha*(delta*trace[i] - Qw*features1[j]);
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
		const int dim)
{

	int i= get_global_id(0);
	int size= get_global_size(0);

	float rho= rhoArray[i];
	float reward= rewardArray[i];
	float gamma= gammaArray[i];

	//Compute the TD error
	float2 delta= computeDeltaGTD(theta, features1, features2, gamma, reward, i, dim, size);

	//update the prediction
	prediction[i]= delta.y;

	//Update the elligibility trace
	updateTraceGTD(trace, features1, rho, gamma, LAMBDA, i, dim, size);

	//Update Theta
	updateThetaGTD(theta, w, trace, features1, ALPHA, gamma, delta.x, LAMBDA, i, dim, size);
	
	//Update w
	updateWGTD(w, trace, features1, ALPHA*ETA, delta.x, i, dim, size);

}

/* The same function as updateGTDLambda but without out any function calls
* This might influence performance (or not)
*/
__kernel void
 updateGTDLambdaNoCalls(__global float* theta, 
		__global float* w,
		__global float* trace, 
		__global const float* features1,
		__global const float* features2, 
		__global const float* rhoArray, 
		__global const float* rewardArray,
		__global const float* gammaArray,
		__global float* prediction,
		const int dim)
{

	int index= get_global_id(0);
	int numDemons= get_global_size(0);

	int i;
	int j;

	float gamma= gammaArray[index];

	//Compute the TD error
	j=0;
	float Q1=0.0f, Q2=0.0f;
	float delta;
	for(i=index; i<dim*numDemons; i+= numDemons){
		Q1 += theta[i]*features1[j];
		Q2 += theta[i]*features2[j];
		j++;
	}
	delta= rewardArray[index] + gamma*Q2 - Q1;


	//update the prediction
	prediction[i]= Q2;

	//Update the elligibility trace
	j=0;
	for(i=index; i<dim*numDemons; i+= numDemons){
		trace[i]= rhoArray[index]*(features1[j] + gamma*LAMBDA*trace[i]);
		j++;
	}

	//Update Theta
	j=0;
	float one_minus_lambda= 1.0f-LAMBDA;
	for(i=index; i<dim*numDemons; i+= numDemons){
		Q1 += trace[i]*w[i];
	}
	for(i=index; i<dim*numDemons; i+= numDemons){
		theta[i] = theta[i] + ALPHA*(delta*trace[i] 
				- gamma*(one_minus_lambda)*Q1*features1[j]);
		j++;
	}
	
	//Update w
	j=0;
	Q1=0.0f;
	for(i=index; i<dim*numDemons; i+= numDemons){
		Q1 += features1[j]*w[i];
		j++;
	}
	j=0;
	for(i=index; i<dim*numDemons; i+= numDemons){
		w[i] = w[i] + ALPHA*ETA*(delta*trace[i] - Q1*features1[j]);
		j++;
	}

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
		theta[i]=0.0f;
		w[i]=0.0f;
		trace[i]=0.0f;
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
__kernel void predict(__global float* theta,
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

