/*
 * helper_functions.h
 * Some helper functions for the 2D particle filter.
 *  Created on: Dec 13, 2016
 *      Author: Tiffany Huang
 */

#ifndef HELPER_FUNCTIONS_H_
#define HELPER_FUNCTIONS_H_

#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "map.h"

using namespace std;

/*
 * Struct representing one position/control measurement.
 */
struct control_s 
{
	
	double velocity;	/* Velocity [m/s] */
	double yawrate;		/* Yaw rate [rad/s] */
};

/*
 * Struct representing one ground truth position.
 */
struct ground_truth 
{
	
	double x;		/* Global vehicle x position [m] */
	double y;		/* Global vehicle y position */
	double theta;	/* Global vehicle yaw [rad] */
};

/*
 * Struct representing one landmark observation measurement.
 */
struct LandmarkObs {
	
	int id;				/* Id of matching landmark in the map. */
	double x;			/* Local (vehicle coordinates) x position of landmark observation [m] */
	double y;			/* Local (vehicle coordinates) y position of landmark observation [m] */

	/* Simple overload to sort landmark values by distance */
	bool operator < (const LandmarkObs& other) const 
	{
		/* Simply compare the euclidian distance to origin */
		return (sqrt(x * x + y * y) < sqrt(other.x * other.x + other.y * other.y));
	}
};

/*
 * Computes the Euclidean distance between two 2D points
 *
 * @param (x1,y1) x and y coordinates of first point
 * @param (x2,y2) x and y coordinates of second point
 * @output Euclidean distance between two 2D points
 */
inline double dist(double x1, double y1, double x2, double y2) 
{
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

/*
 * Computes the error between the ground truth and the predicted values
 *
 * @param (gt_x, gt_y, gt_theta) Ground truth x, y and theta values
 * @param (pf_x, pf_y, pf_theta) Predicted x, y and theta values
 * @output Absolute error between ground truth and predicted values
 */
inline double * getError(double gt_x, double gt_y, double gt_theta, double pf_x, double pf_y, double pf_theta) 
{
	static double error[3];
	error[0] = fabs(pf_x - gt_x);
	error[1] = fabs(pf_y - gt_y);
	error[2] = fabs(pf_theta - gt_theta);
	return error;
}

/* Reads map data from a file.
 * @param filename Name of file containing map data.
 * @output True if opening and reading file was successful
 */
inline bool read_map_data(std::string filename, Map& map) {

	/* Get file of map */
	std::ifstream in_file_map(filename.c_str(),std::ifstream::in);

	/* Return if we can't open the file */
	if (!in_file_map) 
		return false;
	
	/* Declare single line of map file */
	std::string line_map;

	/* Run over each single line */
	while(getline(in_file_map, line_map))
	{
		std::istringstream iss_map(line_map);
 
		/* Declare landmark values and ID */
		double landmark_x_f, landmark_y_f;
		int id_i;

		/* Read data from current line to values */
		iss_map >> landmark_x_f;
		iss_map >> landmark_y_f;
		iss_map >> id_i;

		/* Declare single_landmark */
		Map::single_landmark_s single_landmark_temp;

		/* Set values */
		single_landmark_temp.id_i = id_i;
		single_landmark_temp.x_f  = landmark_x_f;
		single_landmark_temp.y_f  = landmark_y_f;

		/* Add to landmark list of map */
		map.landmark_list.push_back(single_landmark_temp);
	}

	return true;
}

/* Reads control data from a file.
 * @param filename Name of file containing control measurements.
 * @output True if opening and reading file was successful
 */
inline bool read_control_data(std::string filename, std::vector<control_s>& position_meas) 
{

	/* Get file of position measurements */
	std::ifstream in_file_pos(filename.c_str(),std::ifstream::in);

	/* Return if we can't open the file */
	if (!in_file_pos) 
		return false;

	/* Declare single line of position measurement file */
	std::string line_pos;

	/* Run over each single line */
	while(getline(in_file_pos, line_pos))
	{

		std::istringstream iss_pos(line_pos);

		/* Declare position values */
		double velocity, yawrate;

		/* Declare single control measurement */
		control_s meas;

		/*read data from line to values*/
		iss_pos >> velocity;
		iss_pos >> yawrate;

		/* Set values */
		meas.velocity = velocity;
		meas.yawrate = yawrate;

		/* Add to list of control measurements */
		position_meas.push_back(meas);
	}

	return true;
}

/* Reads ground truth data from a file.
 * @param filename Name of file containing ground truth.
 * @output True if opening and reading file was successful
 */
inline bool read_gt_data(std::string filename, std::vector<ground_truth>& gt) 
{

	/* Get file of position measurements */
	std::ifstream in_file_pos(filename.c_str(),std::ifstream::in);

	/* Return if we can't open the file */
	if (!in_file_pos)
		return false;

	/* Declare single line of position measurement file */
	std::string line_pos;

	/* Run over each single line */
	while(getline(in_file_pos, line_pos))
	{
		std::istringstream iss_pos(line_pos);

		/* Declare position values */
		double x, y, azimuth;

		/* Declare single ground truth */
		ground_truth single_gt; 

		/* Read data from line to values */
		iss_pos >> x;
		iss_pos >> y;
		iss_pos >> azimuth;

		/* Set values */
		single_gt.x = x;
		single_gt.y = y;
		single_gt.theta = azimuth;

		/* Add to list of control measurements and ground truth */
		gt.push_back(single_gt);
	}

	return true;
}

/* Reads landmark observation data from a file.
 * @param filename Name of file containing landmark observation measurements.
 * @output True if opening and reading file was successful
 */
inline bool read_landmark_data(std::string filename, std::vector<LandmarkObs>& observations) 
{

	/* Get file of landmark measurements */
	std::ifstream in_file_obs(filename.c_str(),std::ifstream::in);

	/* Return if we can't open the file */
	if (!in_file_obs)
		return false;

	/* Declare single line of landmark measurement file */
	std::string line_obs;

	/* Run over each single line */
	while(getline(in_file_obs, line_obs))
	{
		std::istringstream iss_obs(line_obs);

		/* Declare position values */
		double local_x, local_y;

		/* Read data from line to values */
		iss_obs >> local_x;
		iss_obs >> local_y;

		/* Declare single landmark measurement */
		LandmarkObs meas;

		/* Set values */
		meas.x = local_x;
		meas.y = local_y;

		/* Add to list of control measurements */
		observations.push_back(meas);
	}

	return true;
}

/*
 * Computes the value of the bivariate gaussian centered at 'center' with dimensional
 *   standard deviation `sigmas` at the given point. For this project, it is
 *   stated that the dimensions are not correlated.
 * @param center The x and y coordinates of the center of the distribution
 * @param point  The x and y coordinates for the point to be calculated
 * @param sigmas The x and y standard deviations for the distribution
 * @output The bivariate gaussian value at 'point'
 */
inline double bivariate_gausian(double center[], double point[], double sigmas[]) 
{
	/* Define human readable terms */
	double rho   = 0; /* Correlation between x and y, as suggested in the lectures */
	double mu_x   = center[0];
	double mu_y   = center[1];
	double x     = point[0];
	double y     = point[1];
	double sig_x = sigmas[0];
	double sig_y = sigmas[1];

	/* Implementing the forumla specified in 
	   https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Non-degenerate_case 

	   Since rho is 0, correlation between x & y, most of the terms here will evaluate to 0
	*/
	/* Pre compute basic values */
	double x_diff = (x - mu_x);
	double y_diff = (y - mu_y);
	double sig_xy = (sig_x * sig_y);
	double rho_2 = rho * rho;

	/* Pre compute terms */
	double scaler_term = 1.0d / (2.0d * M_PI * sig_xy * sqrt(1.0d - rho_2));
	double x_term = (x_diff * x_diff) / (sig_x * sig_x);
	double y_term = (y_diff * y_diff) / (sig_y * sig_y);
	double correlation_term = (2.0d * rho * x_diff * y_diff) / sig_xy;

	/* Compute the bivariate gaussian value at point */
	return scaler_term * exp(-1.0d / (2.0d * (1.0d - rho_2)) * (x_term + y_term - correlation_term));
}

#endif /* HELPER_FUNCTIONS_H_ */
