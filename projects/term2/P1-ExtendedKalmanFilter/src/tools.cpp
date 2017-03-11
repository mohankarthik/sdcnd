#include <iostream>
#include "tools.h"

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	/* check the validity of the following inputs:
	 * the estimation vector size should not be zero
	 * the estimation vector size should equal ground truth vector size */
	if(estimations.size() != ground_truth.size() || estimations.size() == 0)
	{
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	/* Accumulate squared residuals */
	for(unsigned int i=0; i < estimations.size(); ++i)
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
	rmse = rmse.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) 
{
	MatrixXd oResult(3,4);

	/* Recover state parameters */
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	/* Pre-compute a set of terms to avoid repeated calculation */
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	/* Check division by zero */
	if(fabs(c1) < 0.0001)
	{
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return oResult;
	}

	/* Compute the Jacobian matrix */
	oResult << (px/c2), 				(py/c2), 				0, 		0,
		  	   -(py/c1), 				(px/c1), 				0, 		0,
		  	   py*(vx*py - vy*px)/c3, 	px*(px*vy - py*vx)/c3, 	px/c2, 	py/c2;

	return oResult;
}
