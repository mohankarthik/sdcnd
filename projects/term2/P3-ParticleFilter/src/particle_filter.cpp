/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */
/*========== INCLUDES ==========*/
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

/* ================================ DEFINES =================================*/
/*! Minimum required samples to pass the test */
#define NUM_PARTICLES (150)

/*=============================== PROTOTYPES ================================*/
static void transformLandmarks(vector<LandmarkObs>& transformed_landmarks, 
	const Particle& particle, const Map& map_landmarks);
static double bivariate_gausian(double center[], double point[], double sigmas[]);

/* ============================== DATA TYPES =============================== */
/* Simple overload to sort landmark values by distance */
struct landmark_comp 
{
    inline bool operator () (const LandmarkObs& A, const LandmarkObs& B) 
	{
		/* Simply compare the euclidian distance to origin */
		return (sqrt((A.x * A.x) + (A.y * A.y)) < sqrt((B.x * B.x) + (B.y * B.y)));
	}
};

/* ================================= CODE ================================== */
/**
 * init Initializes particle filter by initializing particles to Gaussian
 *   distribution around first position and all the weights to 1.
 * 
 * @param x Initial x position [m] (simulated estimate from GPS)
 * @param y Initial y position [m]
 * @param theta Initial orientation [rad]
 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
 *   standard deviation of yaw [rad]]
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
	/* Create a distribution for x, y and theta */
	default_random_engine gen;
	normal_distribution<double> x_distribution(x, std[0]);
	normal_distribution<double> y_distribution(y, std[1]);
	normal_distribution<double> theta_distribution(theta, std[2]);

	/* Loop around and initialize each particle */
	num_particles = NUM_PARTICLES;
	for (uint32_t nLC = 0u; nLC < NUM_PARTICLES; nLC++) 
	{
		Particle particle;
		particle.x = x_distribution(gen);
		particle.y = y_distribution(gen);
		particle.theta = theta_distribution(gen);
		particle.weight = 1.0d;
		particles.push_back(particle);
	}

	/* Set the size of the weight array */
	weights.resize(NUM_PARTICLES);

	/* Set the flag */
	is_initialized = true;
}

/**
 * prediction Predicts the state for the next time step
 *   using the process model.
 * 
 * @param delta_t Time between time step t and t+1 in measurements [s]
 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
 *   standard deviation of yaw [rad]]
 * @param velocity Velocity of car from t to t+1 [m/s]
 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	default_random_engine gen;

	normal_distribution<double> x_noise(0, std_pos[0]);
	normal_distribution<double> y_noise(0, std_pos[1]);
	normal_distribution<double> theta_noise(0, std_pos[2]);

	/* Loop through all particles */
	for (uint32_t nLC = 0u; nLC < NUM_PARTICLES; nLC++) 
	{
		/* Update the particle based on the motion model (including the noise) */
		float delta_theta = particles[nLC].theta + (yaw_rate * delta_t);
		float multiplier   = velocity / yaw_rate;
		particles[nLC].x += multiplier * (sin(delta_theta) - sin(particles[nLC].theta)) + x_noise(gen);
		particles[nLC].y += multiplier * (cos(particles[nLC].theta) - cos(delta_theta)) + y_noise(gen);
		particles[nLC].theta = delta_theta + theta_noise(gen);
	}
}

/**
 * Updates the weights for each particle based on the likelihood of the 
 *   observed measurements. 
 *
 * @param sensor_range Range [m] of sensor
 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
 *   standard deviation of bearing [rad]]
 * @param observations Vector of landmark observations
 * @param map Map class containing map landmarks
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) 
{
	/* Loop through all particles */
	for (uint32_t nLC = 0u; nLC < NUM_PARTICLES; nLC++) 
	{
		Particle particle = particles[nLC];

		/* Transform the map landmarks to the particle coordinate system putting
		 * the particle at the origin oriented toward the positive y-axis */
		vector<LandmarkObs> transformed_landmarks;
		transformLandmarks(transformed_landmarks, particle, map_landmarks);

		/* LandmarkObs are sorted by their distance from the origin */
		sort(transformed_landmarks.begin(), transformed_landmarks.end(), landmark_comp());

		/* Compute the posterior probability given the landmarks */
		particle.weight = 1.0d;
		for (uint32_t nLC1 = 0; nLC1 < observations.size(); nLC1++) 
		{
			LandmarkObs prediction = transformed_landmarks[nLC1];
			LandmarkObs observation = observations[nLC1];
			double prediction_point[] = {prediction.x, prediction.y};
			double observed_point[] = {observation.x, observation.y};
			particle.weight *= bivariate_gausian(observed_point, prediction_point, std_landmark);
		}

		/* Copy the values back into the arrays */
		particles[nLC] = particle;
		weights[nLC] = particle.weight;
	}
}

/**
 * Resamples from the updated set of particles to form the new set of particles.
 */
void ParticleFilter::resample(void) 
{
	default_random_engine gen;
	vector<Particle> resampled_particles;

	/* Compute the maximum weight and set the distribution to twice that value */
	double max_weight = *max_element(begin(weights), end(weights));
	uniform_real_distribution<double> beta_uniform_distribution(0, 2 * max_weight);

	/* Initialize the index to an arbitrary point */
	double beta = 0.0d;
	uint32_t index = (rand() % (uint32_t)(NUM_PARTICLES + 1u));
	
	/* Loop through all the particles */
	for (uint32_t nLC = 0u; nLC < NUM_PARTICLES; nLC++) 
	{
		/* Keep moving around the circle, until beta > sum(weight),
           If some beta is left over from the previous cycle, then
           reuse it, don't discard, hence +=
		 */
		beta += beta_uniform_distribution(gen);
		while (weights[index] < beta) 
		{
			beta -= weights[index];
			index = (index + 1u) % num_particles;
		}

		/* Once we find our sample, copy it (excluding the id) */
		Particle sample = particles[index];
		Particle particle;
		particle.x = sample.x;
		particle.y = sample.y;
		particle.theta = sample.theta;
		particle.weight = sample.weight;

		/* Add it to the list */
		resampled_particles.push_back(particle);
	}

	particles = resampled_particles;
}

/*
 * Writes particle positions to a file.
 *
 * @param filename File to write particle positions to.
 */
void ParticleFilter::write(std::string filename) 
{
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) 
	{
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

/* ====================== STATIC FUNCTIONS =======================*/
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
static void transformLandmarks(vector<LandmarkObs>& transformed_landmarks, 
	const Particle& particle, const Map& map_landmarks) 
{
	/* Loop through all the landmarks */
	for (uint32_t nLC = 0; nLC < map_landmarks.landmark_list.size(); nLC++) 
	{
		/* Get a reference to this landmark */
		const Map::single_landmark_s& landmark = map_landmarks.landmark_list[nLC];

		/* Compute the sin and cos co-efficients */
		double cos_theta = cos(particle.theta - M_PI / 2);
		double sin_theta = sin(particle.theta - M_PI / 2);

		/* Define the resulting landmark */
		LandmarkObs transformed_landmark;
		transformed_landmark.id = landmark.id_i;
		
		/* Compute the new landmark */
		transformed_landmark.x = -(landmark.x_f - particle.x) * sin_theta + (landmark.y_f - particle.y) * cos_theta;
		transformed_landmark.y = -(landmark.x_f - particle.x) * cos_theta - (landmark.y_f - particle.y) * sin_theta;

		/* Add it to the return list */
		transformed_landmarks.push_back(transformed_landmark);
	}
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
static double bivariate_gausian(double center[], double point[], double sigmas[])
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
