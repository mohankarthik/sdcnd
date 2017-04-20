/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

/* ==================== INCLUDES ==================== */
#include <vector>
#include "helper_functions.h"

using namespace std;

/* =================== DATA TYPES =================== */
/*! Structure that holds a definition of a single particle */
struct Particle 
{
	int id; /*!< ID of the particle */
	double x; /*!< The x position */
	double y; /*!< The y position */
	double theta; /*!< The current heading */ 
	double weight; /*!< The weight of the particle */
};

/* ================ CLASS DEFINITION ================ */
/*! Defnition of the Particle Filter class */
class ParticleFilter 
{
	/*! Number of particles to draw */
	int num_particles; 
		
	/*! Flag, if filter is initialized */
	bool is_initialized;
	
	/*! Vector of weights of all particles */
	std::vector<double> weights;
	
public:
	
	/*! Set of current particles */
	std::vector<Particle> particles;

	/* Constructor
	 * @param M Number of particles
	 */
	ParticleFilter() : num_particles(0), is_initialized(false) {}

	/* Destructor */
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);
	

	/**
	 * Transforms all the landmarks with respect to this particular particle's 
	 *    co-ordinate system.
	 *
	 * @param [out] transformed_landmarks Vector of landmarks that are now in 
	 *   the particle's co-ordinate system
	 * @param [in] particle The particle whose coordinate system we want to convert the
	 *   landmarks to
	 * @param [in] map_landmarks All the landmarks
	 */
	void transformLandmarks(vector<LandmarkObs>& transformed_landmarks, 
			const Particle& particle, const Map& map_landmarks) ;

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
	 *   standard deviation of bearing [rad]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations,
			Map map_landmarks);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample(void);
	
	/*
	 * write Writes particle positions to a file.
	 * @param filename File to write particle positions to.
	 */
	void write(std::string filename);
	
	/**
	 * initialized Returns whether particle filter is initialized yet or not.
	 */
	const bool initialized() const 
	{
		return is_initialized;
	}
};

#endif /* PARTICLE_FILTER_H_ */
