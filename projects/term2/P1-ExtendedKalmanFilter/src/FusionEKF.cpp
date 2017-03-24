#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/* Empirically set the acceleration noise */
#define ACCEL_NOISE (9u)

/* Measurement noise covariance values */
#define STD_LASER_X (0.0225f)
#define STD_LASER_Y (0.0225f)

#define STD_RADAR_RHO (0.09f)
#define STD_RADAR_THE (0.0009f)
#define STD_RADAR_RD  (0.09f)

/*
 * Constructor.
 */
FusionEKF::FusionEKF() 
{
  /* Initialize the flags */
  is_initialized_ = false;
  previous_timestamp_ = 0ll;

  /* Set the noise values */
  noise_ax = ACCEL_NOISE;
  noise_ay = ACCEL_NOISE;

  /* Initialize the laser measurement noise matrix */
  R_laser_ = MatrixXd(2u, 2u);
  R_laser_ << STD_LASER_X, 0.0f,
              0.0f, STD_LASER_Y;
  
  /* Initialize the radar measurement noise matrix */
  R_radar_ = MatrixXd(3u, 3u);
  R_radar_ << STD_RADAR_RHO, 0.0f, 0.0f,
              0.0f, STD_RADAR_THE, 0.0f,
              0.0f, 0.0f, STD_RADAR_RD;

  /* Initialize the laser measurement function,
   * we can measure only the px and py 
   */
  H_laser_ = MatrixXd(2u, 4u);
  H_laser_ << 1.0f, 0.0f, 0.0f, 0.0f,
              0.0f, 1.0f, 0.0f, 0.0f;

  /* Create the Measurement matrix H for radar 
   * this will be filled with a Jacobian approximation
   * of the nonlinear function later
   */
  H_radar_ = MatrixXd(3u, 4u);

  /* Create the state transition matrix F */
  ekf_.F_ = MatrixXd(4u, 4u);

  /* Initialize the state covariance matrix
   * such that we have very high covariance (uncertainity)
   * in the velocities, and pretty low on the positions */
  ekf_.P_ = MatrixXd(4u, 4u);
  ekf_.P_ << 0.1f, 0.0f, 0.0f, 0.0f,
             0.0f, 0.1f, 0.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f;

  /* Create the state matrix */
  ekf_.x_ = VectorXd(4u);

  /* Create the noise covariance matrix */
  ekf_.Q_ = MatrixXd(4u, 4u);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) 
{
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  /* If this is the first time we are measuring */
  if (is_initialized_ == false) 
  {
    /* Initialize the features */
    ekf_.x_ = VectorXd(4u);
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
    {
      ekf_.x_ << measurement_pack.raw_measurements_[0u] * cos(measurement_pack.raw_measurements_[1u]), 
                 measurement_pack.raw_measurements_[0u] * sin(measurement_pack.raw_measurements_[1u]), 
                 0.0f, 
                 0.0f;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) 
    {
      ekf_.x_ << measurement_pack.raw_measurements_[0u], 
                 measurement_pack.raw_measurements_[1u], 
                 0.0f, 
                 0.0f;
    }

    /* Save as the previous timestamp, this is to get a delta time of 0 in the first measurement */
    previous_timestamp_ = measurement_pack.timestamp_;

    /* Set the flag */
    is_initialized_ = true;

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  /* Calculate the time diff */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0f;
  previous_timestamp_ = measurement_pack.timestamp_;

  /* If the time differnce is too less, we need not predict again */
  if (fabs(dt) > 0.000001f)
  {
    /* Update the state transition with the time diff */
    ekf_.F_ << 1.0f, 0.0f, dt  , 0.0f,
               0.0f, 1.0f, 0.0f, dt,
               0.0f, 0.0f, 1.0f, 0.0f,
               0.0f, 0.0f, 0.0f, 1.0f;

    /* Update the modelling error covariance matrix based on the accellartion noises */
    float dt_2 = pow(dt, 2.0f);
    float dt_3 = pow(dt, 3.0f);
    float dt_4 = pow(dt, 4.0f);
    ekf_.Q_ << 0.25f * dt_4 * noise_ax, 0.0f, 0.5f * dt_3 * noise_ax, 0.0f,
               0.0f, 0.25f * dt_4 * noise_ay, 0.0f, 0.5f * dt_3 * noise_ay,
               0.5f * dt_3 * noise_ax, 0.0f, dt_2 * noise_ax, 0.0f,
               0.0f, 0.5f * dt_3 * noise_ay, 0.0f, dt_2 * noise_ay;

    /* Call KalmanFilter to predict */
    ekf_.Predict();
  }

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

    /* If the Jacobian is 0, for some reason, don't update */
    if (!ekf_.H_.isZero()) 
    {
      ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    }
  }
  /* Update the laser measurement and sensor covariance matrices */
  else 
  {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  /* Print out the state and covariance */
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
  //cin >> dt;
}
