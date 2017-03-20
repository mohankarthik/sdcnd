#include <iostream>
#include "ukf.h"

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() 
{
  /*
   * Generic initializations
   */
  is_initialized_ = false; /* Set the state that this class isn't initialized yet */
  time_us_ = 0u;
  NIS_radar_ = 0.0f;
  NIS_laser_ = 0.0f;

  /*
   * Configurations
   */
  use_laser_  = true;   /* Use laser measurements or not */
  use_radar_  = true;   /* Use Radar measurements or not */
  std_a_      = 3.0f;   /* Std.dev of the acceleration noise: typically 3m/s^2 
                         * since typical urban max acceleration is 6m/s^2 */
  std_yawdd_  = 30.0f;  /* Std deviation of yaw acceleration process noise */
  std_laspx_  = 0.15f;  /* Std deviation of laser x orientation measurement noise */
  std_laspy_  = 0.15f;  /* Std deviation of laser y orientation measurement noise */
  std_radr_   = 0.3f;   /* Std deviation of radar rho measurement noise */
  std_radphi_ = 0.03f;  /* Std deviation of radar phi measurement noise */
  std_radrd_  = 0.3f;   /* Std deviation of radar rho change measurement noise */
  
  /*
   * CTRV configurations
   */
  n_x_    = 5u; /* Number of state variables */
  n_aug_  = 7u; /* Number of augmented state variables */
  n_sig_  = (2u * n_aug_) + 1u; /* Number of sigma points */
  lambda_ = 3u - n_aug_; /* Value of lamdba parameter */

  /*
   * Matrices
   */
  Xsig_pred_ = MatrixXd(n_x_, n_sig_); /* Predicted sigma points initialized to 0 */
  Xsig_pred_.fill(0.0f);

  weights_ = VectorXd(n_sig_); /* weights_ for the sigma point operations */
  weights_(0u) = lambda_ / (lambda_ + n_aug_);
  for(unsigned int i = 1u; i < n_sig_; i++)
  {  
    weights_(i) = 0.5f / (lambda_ + n_aug_);
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
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
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0f;
    time_us_= meas_package.timestamp_;

    /* Predict the posterior */
    try 
    {
      Prediction(delta_t);
    } 
    catch (std::range_error e) 
    {
      std::cerr << "Prediction failed, restarting the filter" << std::endl;

      /* If convariance matrix is non positive definite (because of numerical instability),
       * restart the filter using previous measurement as initialiser. */
      InitializeMeasurement(prev_meas_package_);

      /* Redo prediction using the current measurement
       * We don't get exception this time, because initial P (identity) is positive definite. */
      Prediction(delta_t);
    }

    /* Update the posterior with the measurement */
    if(meas_package.sensor_type_==MeasurementPackage::LASER)
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
  x_aug(n_x_) = 0.0f;
  x_aug(n_x_ + 1u) = 0.0f;

  /* Create augmented covariance matrix */
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0f);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1u, n_x_ + 1u) = std_yawdd_ * std_yawdd_;

  /* Take matrix square root */
  /* 1. compute the Cholesky decomposition of P_aug */
  Eigen::LLT<MatrixXd> lltOfPaug(P_aug);
  if (lltOfPaug.info() == Eigen::NumericalIssue) 
  {
    /* if decomposition fails, we have numerical issues */
    throw std::range_error("LLT failed");
  }
  /* 2. get the lower triangle */
  MatrixXd L = lltOfPaug.matrixL();

  /* create augmented sigma points */
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  Xsig_aug.col(0u)  = x_aug;
  for (unsigned int i = 0u; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1u)          = x_aug + (sqrt(lambda_+n_aug_) * L.col(i));
    Xsig_aug.col(i + 1u + n_aug_) = x_aug - (sqrt(lambda_+n_aug_) * L.col(i));
  }
  
  /* Predict sigma points */
  for (unsigned int i = 0u; i < n_sig_; i++)
  {
    /* extract values for better readability */
    double px = Xsig_aug(0u,i);
    double py = Xsig_aug(1u,i);
    double v = Xsig_aug(2u,i);
    double yaw = Xsig_aug(3u,i);
    double yawd = Xsig_aug(4u,i);
    double nu_a = Xsig_aug(5u,i);
    double nu_yawdd = Xsig_aug(6u,i);

    /* Predict the values */
    double px_p, py_p;
    if (abs(yawd) > 0.001f) 
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

    /* add noise */
    px_p = px_p + (0.5f * nu_a * delta_t * delta_t * cos(yaw));
    py_p = py_p + (0.5f * nu_a * delta_t * delta_t * sin(yaw));
    v_p = v_p + (nu_a * delta_t);
    yaw_p = yaw_p + (0.5f * nu_yawdd * delta_t * delta_t);
    yawd_p = yawd_p + (nu_yawdd * delta_t);

    /* write predicted sigma point into the correct column */
    Xsig_pred_(0u,i) = px_p;
    Xsig_pred_(1u,i) = py_p;
    Xsig_pred_(2u,i) = v_p;
    Xsig_pred_(3u,i) = yaw_p;
    Xsig_pred_(4u,i) = yawd_p;
  }

  /* Compute the predicted state's mean */
  x_.fill(0.0f);
  for (unsigned int i = 0u; i < n_sig_; i++) 
  {
    x_ = x_ + (weights_(i) * Xsig_pred_.col(i));
  }

  /* Compute the predicted state's covariance */
  P_.fill(0.0f);
  for (unsigned int i = 0u; i < n_sig_; i++) 
  {
    /* state difference */
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    /* angle normalization */
    while(x_diff(3u) > M_PI)
    {
      x_diff(3u) -= (2.0f * M_PI);
    }
    while(x_diff(3u) < -M_PI)
    {
      x_diff(3u) += (2. * M_PI);
    } 

    P_ = P_ + (weights_(i) * x_diff * x_diff.transpose());
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}

/******* PRIVATE FUNCTIONS ***********/
/**
 * Intializes the state and covariance matrices based on the measurement
 * @param {measurementPackage} meas_package
 */
void UKF::InitializeMeasurement(MeasurementPackage meas_package)
{
  x_ = VectorXd(n_x_); /* The state vector X */

  /* Initialize the state to the measurements itself */
  if(meas_package.sensor_type_==MeasurementPackage::LASER)
  {
    x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0f, 0.0f, 0.0f;
  }
  else 
  {
    double  meas_rho =  meas_package.raw_measurements_[0];
    double meas_phi =  meas_package.raw_measurements_[1];
    x_ << meas_rho * cos(meas_phi), meas_rho * sin(meas_phi), 0.0f, 0.0f, 0.0f;
  }

  /* Initialize the covariance matrix */
  P_ = MatrixXd(n_x_, n_x_);
  VectorXd diag(5);
  diag << 1.0f, 1.0f, 1000.0f, M_PI_2, 0.1f; /* Very low confidence in psi. High confidence in psi dot */
  P_ = diag.asDiagonal();

  /* Set the time */
  time_us_ = meas_package.timestamp_;
}
