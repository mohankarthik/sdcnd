#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>
#include <stdlib.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define FLOAT_THRESHOLD (0.0001d)

/**
 * Initializes Unscented Kalman filter
 * @param {bool} enable_lidar: Enables or disables Lidar measurement updates
 * @param {bool} enable_radar: Enables or disables Radar measurement updates
 * @param {bool} dbg_enabled: If we should output x and P values at each step
 */
UKF::UKF(bool enable_lidar, bool enable_radar, bool dbg_enabled) 
{ 
  /*
   * Generic initializations
   */
  is_initialized_ = false; /* Set the state that this class isn't initialized yet */
  time_us_ = 0ll;
  NIS_radar_ = 0.0d;
  NIS_laser_ = 0.0d;

  /*
   * Configurations
   */
  dbg_enabled_ = dbg_enabled; /* Store the diagnostics file */
  use_laser_  = enable_lidar;   /* Use laser measurements or not */
  use_radar_  = enable_radar;   /* Use Radar measurements or not */
  std_a_      = 0.4d;   /* Std.dev of the acceleration noise: typically 3m/s^2 
                         * since typical urban max acceleration is 6m/s^2 */
  std_yawdd_  = 0.4d;   /* Std deviation of yaw acceleration process noise */
  std_laspx_  = 0.015d;  /* Std deviation of laser x orientation measurement noise */
  std_laspy_  = 0.015d;  /* Std deviation of laser y orientation measurement noise */
  std_radr_   = 0.5d;   /* Std deviation of radar rho measurement noise */
  std_radphi_ = 0.01d;  /* Std deviation of radar phi measurement noise */
  std_radrd_  = 0.25d;   /* Std deviation of radar rho change measurement noise */
  
  /*
   * CTRV configurations
   */
  n_x_    = 5; /* Number of state variables */
  n_aug_  = 7; /* Number of augmented state variables */
  n_sig_  = (2 * n_aug_) + 1; /* Number of sigma points */
  lambda_ = 3 - n_aug_; /* Value of lamdba parameter */

  /*
   * Measurement model configurations
   */
  n_z_radar_ = 3; /* Number of radar measurement dimensions; rho, thetha and rho dot */
  n_z_lidar_ = 2; /* Number of lidar measurement dimensions; x and y */

  /*
   * Matrices
   */
  Xsig_pred_ = MatrixXd(n_x_, n_sig_); /* Predicted sigma points initialized to 0 */
  Xsig_pred_.fill(0.0d);

  weights_ = VectorXd(n_sig_);
  weights_.segment(1, 2 * n_aug_).fill(0.5d / (n_aug_ + lambda_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  /* Update the process noise */
  Q_ = MatrixXd(2, 2);
  Q_ << std_a_ * std_a_, 0.0d,
        0.0d, std_yawdd_ * std_yawdd_;

  /* The Laser measurement noise */
  R_laser_ = MatrixXd(n_z_lidar_, n_z_lidar_);
  R_laser_ << std_laspx_, 0.0d,
              0.0d, std_laspy_;

  /* The radar measurement noise */
  R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
  R_radar_ << std_radr_ * std_radr_, 0.0d, 0.0d,
              0.0d, std_radphi_ * std_radphi_, 0.0d,
              0.0d, 0.0d, std_radrd_ * std_radrd_;

  /* The state vector X */
  x_ = VectorXd(n_x_);

  /* The augmented sigma points */
  Xsig_ = MatrixXd(n_aug_, n_sig_);

  /* The state covariance matrix */
  P_ = MatrixXd(n_x_, n_x_);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{
  /* Drop the measurement if it's configured to be not used */
  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_ == false)
  {
    return;
  }
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ == false)
  {
    return;
  }

  /* If this is the first measurement */
  if(is_initialized_ == false)
  {
    /* Initialize the state and covariance */
    InitializeMeasurement(meas_package);

    /* Set the flag */
    is_initialized_ = true;
  }
  else
  {
    /* Calculate the time difference between current time and previous measurement */
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0d;
    time_us_= meas_package.timestamp_;

    /* Predict the X and P after delta_t time
     * Note: We don't have to worry about delta_t = 0, and more importantly, we should not skip predictions if delta_t = 0 since
     * we need to regenerate the sigma points after each measurement update. Or the second measurement would use old predicted
     * sigma points
     */
    Prediction(delta_t);

    /* Update the posterior with the measurement */
    if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      UpdateLidar(meas_package);
    }
    else 
    {
      UpdateRadar(meas_package);
    }
  }

  /* Save the measurement package */
  prev_meas_package_ = meas_package;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) 
{
  /* Create augmented mean state */
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0.0d;
  x_aug(n_x_ + 1) = 0.0d;

  /* Create augmented covariance matrix */
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0f);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(Q_.rows(), Q_.cols()) = Q_;

  /* Calculate square root of P */
  MatrixXd A = P_aug.llt().matrixL();

  /* Create augmented sigma points */
  Xsig_.colwise() = x_aug;
  MatrixXd offset = A * sqrt(lambda_ + n_aug_);

  Xsig_.block(0, 1, n_aug_, n_aug_) += offset;
  Xsig_.block(0, n_aug_ + 1, n_aug_, n_aug_) -= offset;
  
  /* Predict sigma points */
  for (int i = 0; i < n_sig_; i++)
  {
    /* Extract values for better readability */
    double px = Xsig_(0,i);
    double py = Xsig_(1,i);
    double v = Xsig_(2,i);
    double yaw = Xsig_(3,i);
    double yawd = Xsig_(4,i);
    double nu_a = Xsig_(5,i);
    double nu_yawdd = Xsig_(6,i);

    /* Predict the values */
    double px_p, py_p;
    if (fabs(yawd) > FLOAT_THRESHOLD) 
    {
        px_p = px + ((v / yawd) * (sin(yaw + (yawd * delta_t)) - sin(yaw)));
        py_p = py + ((v / yawd) * (cos(yaw) - cos(yaw + (yawd * delta_t))));
    }
    else 
    {
        px_p = px + (v * delta_t * cos(yaw));
        py_p = py + (v * delta_t * sin(yaw));
    }
    double v_p = v;
    double yaw_p = yaw + (yawd * delta_t);
    double yawd_p = yawd;

    /* Add noise */
    px_p = px_p + (0.5d * nu_a * delta_t * delta_t * cos(yaw));
    py_p = py_p + (0.5d * nu_a * delta_t * delta_t * sin(yaw));
    v_p = v_p + (nu_a * delta_t);
    yaw_p = yaw_p + (0.5d * nu_yawdd * delta_t * delta_t);
    yawd_p = yawd_p + (nu_yawdd * delta_t);

    /* Write predicted sigma point into the correct column */
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    NormalizeAngle(&Xsig_pred_(3,i));
    Xsig_pred_(4,i) = yawd_p;
  }

  /* Compute the predicted state's mean */
  x_.fill(0.0d);
  for (int i = 0; i < n_sig_; i++) 
  {
    x_ = x_ + (weights_(i) * Xsig_pred_.col(i));
  }

  /* Compute the predicted state's covariance */
  P_.fill(0.0d);
  for (int i = 0; i < n_sig_; i++) 
  {
    /* state difference */
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    /* Angle normalization */
    NormalizeAngle(&x_diff(3));

    /* Compute the covariance */
    P_ = P_ + (weights_(i) * x_diff * x_diff.transpose());
  }

  /* Print out the diagnostics */
  if (dbg_enabled_ == true)
  {
    cout << "---- PREDICTION ----" << endl;
    cout << "Time Diff: " << delta_t << endl;
    cout << "x: " << endl << x_ << endl;
    cout << "P: " << endl << P_ << endl << endl;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  MatrixXd Zsig = MatrixXd(n_z_lidar_, n_sig_);
  VectorXd z_pred = VectorXd(n_z_lidar_);

  /* Just copy over the rows */
  for(int i = 0; i < n_z_lidar_; i++)
  {
    Zsig.row(i) = Xsig_pred_.row(i);
  }

  /* Compute the difference */  
  z_pred = Zsig * weights_;

  /* Measurement covariance matrix S */
  MatrixXd S = MatrixXd(n_z_lidar_, n_z_lidar_);
  S.fill(0.0d);
  for(int i = 0; i < n_sig_; i++) 
  {
    VectorXd residual = Zsig.col(i) - z_pred;
    S = S + (weights_(i) * residual * residual.transpose());
  }

  /* add measurement noise covariance matrix */
  S = S + R_laser_;

  /* Create matrix for cross correlation Tc */
  MatrixXd Tc = MatrixXd(n_x_, n_z_lidar_);
  Tc.fill(0.0d);
  for(int i = 0; i < n_sig_; i++)
  {
    VectorXd tx = Xsig_pred_.col(i) - x_;
    VectorXd tz = Zsig.col(i) - z_pred;
    Tc = Tc + weights_(i)*tx*tz.transpose();
  }

  /* Kalman filter K */
  MatrixXd K = Tc * S.inverse();

  /* residual */
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  /* Update the state and covariance */
  x_ = x_ + (K * z_diff);
  P_ = P_ - (K * S * K.transpose());

  /* Compute the NIS */
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

  /* Print out the diagnostics */
  if (dbg_enabled_ == true)
  {
    cout << "---- LASER MEASUREMENT ----" << endl;
    cout << "Measurements: " << meas_package.raw_measurements_ << endl;
    cout << "x: " << endl << x_ << endl;
    cout << "P: " << endl << P_ << endl;
    cout << "NIS: " << NIS_laser_ << endl << endl;
  }
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) 
{
  MatrixXd Zsig = MatrixXd(n_z_radar_, n_sig_);
  Zsig.fill(0.0d);

  /* transform sigma points into measurement space */
  for (int i = 0; i < n_sig_; i++) 
  {
    /* extract values for better readibility */
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    /* TODO:
     * There is a possible divide by 0 here, but I'm not sure
     * on what to substitute if we do encounter a divide by 0.
     */
    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;
    double sqrt_term = sqrt((p_x * p_x) + (p_y * p_y));

    /* Measurement model */
    Zsig(0,i) = sqrt_term;
    Zsig(1,i) = atan2(p_y, p_x);
    Zsig(2,i) = ((p_x * v1) + (p_y * v2)) / sqrt_term;
  }

  /* mean predicted measurement */
  VectorXd z_pred = VectorXd(n_z_radar_);
  z_pred.fill(0.0d);
  for (int i = 0; i < n_sig_; i++)
  {
      z_pred = z_pred + (weights_(i) * Zsig.col(i));
  }

  /* Measurement covariance matrix S */
  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
  S.fill(0.0d);
  for (int i = 0; i < n_sig_; i++)
  {
    /* Residual */
    VectorXd z_diff = Zsig.col(i) - z_pred;

    /* angle normalization */
    NormalizeAngle(&z_diff(1));
    
    /* Update the measurement covariance */
    S = S + (weights_(i) * z_diff * z_diff.transpose());
  }

  /* Add measurement noise covariance matrix */
  S = S + R_radar_;

  /* Create & calculate matrix for cross correlation Tc */
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
  Tc.fill(0.0d);
  for (int i = 0; i < n_sig_; i++) 
  {
    /* Residual */
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    /* Angle normalization */
    NormalizeAngle(&z_diff(1));

    /* State difference */
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    /* Angle normalization */
    NormalizeAngle(&x_diff(3));

    /* Compute the cross correlation */
    Tc = Tc + (weights_(i) * x_diff * z_diff.transpose());
  }

  /* Kalman gain K */
  MatrixXd K = Tc * S.inverse();

  /* Residual */
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  /* Angle normalization */
  NormalizeAngle(&z_diff(1));

  /* Update state mean and covariance matrix */
  x_ = x_ + (K * z_diff);
  P_ = P_ - (K * S * K.transpose());

  /* Compute the NIS */
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  /* Print out the diagnostics */
  if (dbg_enabled_ == true)
  {
    cout << "---- RADAR MEASUREMENT ----" << endl;
    cout << "Measurements: " << meas_package.raw_measurements_ << endl;
    cout << "x: " << endl << x_ << endl;
    cout << "P: " << endl << P_ << endl;
    cout << "NIS: " << NIS_radar_ << endl << endl;
  }
}

/******* PRIVATE FUNCTIONS ***********/
/**
 * Intializes the state and covariance matrices based on the measurement
 * @param {measurementPackage} meas_package
 */
void UKF::InitializeMeasurement(MeasurementPackage meas_package)
{
  /* Initialize the covariance */
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1000, 0, 0,
        0, 0, 0, 100, 0,
        0, 0, 0, 0, 1;

  /* Initialize the state */
  double px, py;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) 
  {
    double rho = meas_package.raw_measurements_[0];
    double phi = meas_package.raw_measurements_[1];

    px = rho * cos(phi);
    py = rho * sin(phi);
  }
  else
  {
    px = meas_package.raw_measurements_[0];
    py = meas_package.raw_measurements_[1];
  }

  /* If initial values are zero they will set to an initial guess
   * and the uncertainty will be increased.
   * Initial zeros would cause the algorithm to fail when using only Radar data. */
  if(fabs(px) < FLOAT_THRESHOLD)
  {
      px = FLOAT_THRESHOLD;
  }
  if(fabs(py) < FLOAT_THRESHOLD)
  {
      py = FLOAT_THRESHOLD;
  }

  /* Initialize the state */
  x_ << px, py, 0, 0, 0;

  /* Set the time */
  time_us_ = meas_package.timestamp_;
}

/**
 * Normalizes a given double angle between -Pi to Pi
 * @param {measurementPackage} pValue: Variable to be normalized
 */
void UKF::NormalizeAngle(double *pValue)
{
  if (fabs(*pValue) > M_PI)
  {
    *pValue -= round(*pValue / (2.0d * M_PI)) * (2.0d * M_PI);
  }
}