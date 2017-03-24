#include <iostream>
#include "tools.h"

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) 
{
	VectorXd rmse(4);
	rmse.fill(0.0f);

	/* check the validity of the following inputs:
	 * the estimation vector size should not be zero
	 * the estimation vector size should equal ground truth vector size */
	if((estimations.size() != ground_truth.size()) || (estimations.size() == 0u))
	{
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	/* Accumulate squared residuals */
	for(unsigned int i = 0u; i < estimations.size(); ++i)
	{
		/* Find the difference */
		VectorXd diff = estimations[i] - ground_truth[i];

		/* Coefficient-wise multiplication and add to rmse */
		diff = diff.array() * diff.array();
		rmse += diff;
	}

	/* Get the mean */
	rmse /= estimations.size();
	
	/* Get the squared root */
	return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) 
{
	MatrixXd oResult(3u ,4u);

	/* Recover state parameters */
	float px = x_state(0u);
	float py = x_state(1u);
	float vx = x_state(2u);
	float vy = x_state(3u);

	/* Pre-compute a set of terms to avoid repeated calculation */
	float c1 = (px * px) + (py * py);
	float c2 = sqrt(c1);
	float c3 = (c1 * c2);

	/* Check division by zero */
	if(fabs(c1) < 0.0001f)
	{
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return oResult.setZero();
	}

	/* Compute the Jacobian for the radar measurement */
	oResult << 	(px/c2), 				(py/c2), 				0.0f,	0.0f,
  	   	   		-(py/c1), 				(px/c1), 				0.0f, 	0.0f,
  	   	   		py*(vx*py - vy*px)/c3, 	px*(px*vy - py*vx)/c3, 	px/c2, 	py/c2;

	return oResult;
}
