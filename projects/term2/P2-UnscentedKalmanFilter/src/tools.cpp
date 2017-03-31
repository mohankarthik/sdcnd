#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) 
{
	/* Preconditions */
	if (estimations.size() == 0 || estimations.size() != ground_truth.size()) 
	{
        throw std::invalid_argument( "CalculateRMSE () - Error: Invalid input values." );
    }

  	Eigen::VectorXd rmse(estimations[0].array().size());
	rmse.fill(0.0f);

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
