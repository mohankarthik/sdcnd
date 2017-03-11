#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/* Empirically set the acceleration noise 
     * Value  RMSE 
     * 5      1.36
     * 500    1.14
     * 50     1.01
     * 25     1.06
     * 37     1.02
     * 43     1.02
     * 48     1.01 */
#define ACCEL_NOISE (5)

/* Measurement noise covariance values */
#define MEASUREMENT_COVAR (0.0225)

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  H_radar_ = MatrixXd(3, 4);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  /* If this is the first time we are measuring */
  if (!is_initialized_) {
    /* Initialize all the variables */

    /* Initialize the state transition matrix F */
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;

    /* Initialize the state covariance matrix
     * such that we have very high covariance (uncertainity)
     * in the velocities, and pretty low on the positions */
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1000, 0,
               0, 0, 0, 1000;

    
    noise_ax = ACCEL_NOISE;
    noise_ay = ACCEL_NOISE;

    /* Set the covariance of the measurement noise */
    R_laser_ << MEASUREMENT_COVAR, 0,
                0, MEASUREMENT_COVAR;

    R_radar_ << MEASUREMENT_COVAR, 0, 0,
                0, MEASUREMENT_COVAR, 0,
                0, 0, MEASUREMENT_COVAR;

    /* Set the measurement matrix for the laser
     * specifically, we can only measure the 
     * x and y */
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    /* Set the measurement matrix for the radar,
     * specifically, we measure rho and theta, but also
     * all the doppler, which gives us the combined velocities */
    H_radar_  << 1, 1, 0, 0,
                 1, 1, 0, 0,
                 1, 1, 1, 1;


    /* Initialize the features */
    ekf_.x_ = VectorXd(4);
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
    {
      /* Convert from polar to carterisan in the case of Radar */
      float x_cart = measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]);
      float y_cart = measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]);

      ekf_.x_ << x_cart, y_cart, 0, 0;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) 
    {
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    /* Save as the previous timestamp, this is to get a delta time of 0 in the first measurement */
    previous_timestamp_ = measurement_pack.timestamp_;

    /* Set the flag */
    is_initialized_ = true;

    return;
  }

  /* Calculate the time diff */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  /* Update the state transition with the time diff */
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;

  /* Update the modelling error covariance matrix based on the accellartion noises */
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << (pow(dt,4)/4*noise_ax), 0, (pow(dt,3)/2*noise_ax), 0,
              0, (pow(dt,4)/4*noise_ay), 0, (pow(dt,3)/2*noise_ay),
              (pow(dt,3)/2*noise_ax), 0, pow(dt,2)*noise_ax, 0,
              0, (pow(dt,3)/2*noise_ay), 0, pow(dt,2)*noise_ay;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  /* Call KalmanFilter to predict */
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  /* Update the radar measurement and sensor covariance matrices */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
  {
    /* Compute the jacobian, which will become the measurement matrix */
    H_radar_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = H_radar_;
    ekf_.R_ = R_radar_;
  }
  /* Update the laser measurement and sensor covariance matrices */
  else 
  {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
  }

  /* Update using the kalman filter */
  ekf_.Update(measurement_pack.raw_measurements_);


  /* Print out the state and covariance */
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
