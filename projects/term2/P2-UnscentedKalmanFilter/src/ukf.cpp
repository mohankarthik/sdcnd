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
   * Measurement model configurations
   */
  n_z_radar_ = 3u; /* Number of radar measurement dimensions; rho, thetha and rho dot */
  n_z_lidar_ = 2u; /* Number of lidar measurement dimensions; x and y */

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
void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  MatrixXd Zsig = MatrixXd(n_z_lidar_, n_sig_);
  VectorXd z_pred = VectorXd(n_z_lidar_);

  /* Just copy over the rows */
  for(unsigned int i = 0u; i < n_z_lidar_; i++)
  {
    Zsig.row(i) = Xsig_pred_.row(i);
  }

  /* Compute the difference */  
  z_pred = Zsig * weights_;

  /* Measurement covariance matrix S */
  MatrixXd S = MatrixXd(n_z_lidar_, n_z_lidar_);
  S.fill(0.0f);
  for(unsigned int i = 0u; i < n_sig_; i++) 
  {
    VectorXd residual = Zsig.col(i) - z_pred;
    S = S + (weights_(i) * residual * residual.transpose());
  }

  /* add measurement noise covariance matrix */
  MatrixXd R(n_z_lidar_, n_z_lidar_);
  R <<    std_laspx_*std_laspx_,0.0f,
          0.0f,std_laspy_*std_laspy_;
  S = S + R;

  /* Create matrix for cross correlation Tc */
  MatrixXd Tc = MatrixXd(n_x_, n_z_lidar_);
  Tc.fill(0.0f);
  for(unsigned int i = 0u; i < n_sig_; i++)
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
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) 
{
  MatrixXd Zsig = MatrixXd(n_z_radar_, n_sig_);

  /* transform sigma points into measurement space */
  for (unsigned int i = 0u; i < n_sig_; i++) 
  {
    /* extract values for better readibility */
    double p_x = Xsig_pred_(0u,i);
    double p_y = Xsig_pred_(1u,i);
    double v  = Xsig_pred_(2u,i);
    double yaw = Xsig_pred_(3u,i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;
    double sqrt_term = sqrt((p_x * p_x) + (p_y * p_y));

    /* measurement model */
    Zsig(0u,i) = sqrt_term;
    Zsig(1u,i) = atan2(p_y, p_x);
    Zsig(2u,i) = ((p_x * v1) + (p_y * v2)) / sqrt_term;
  }

  /* mean predicted measurement */
  VectorXd z_pred = VectorXd(n_z_radar_);
  z_pred.fill(0.0f);
  for (unsigned int i = 0u; i < n_sig_; i++)
  {
      z_pred = z_pred + (weights_(i) * Zsig.col(i));
  }

  /* Measurement covariance matrix S */
  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
  S.fill(0.0f);
  for (unsigned int i = 0u; i < n_sig_; i++)
  {
    /* Residual */
    VectorXd z_diff = Zsig.col(i) - z_pred;

    /* angle normalization */
    while(z_diff(1)> M_PI)
    {
      z_diff(1) -= (2.0f * M_PI);
    }
    while(z_diff(1)<-M_PI)
    {
      z_diff(1) += (2.0f * M_PI);
    }

    S = S + (weights_(i) * z_diff * z_diff.transpose());
  }

  /* add measurement noise covariance matrix */
  MatrixXd R = MatrixXd(n_z_radar_, n_z_radar_);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;

  /* Create matrix for cross correlation Tc */
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);

  /* Calculate cross correlation matrix */
  Tc.fill(0.0f);
  for (unsigned int i = 0u; i < n_sig_; i++) 
  {
    /* Residual */
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    /* angle normalization */
    while(z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while(z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    /* state difference */
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    /* angle normalization */
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + (weights_(i) * x_diff * z_diff.transpose());
  }

  /* Kalman gain K */
  MatrixXd K = Tc * S.inverse();

  /* residual */
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  /* update state mean and covariance matrix */
  x_ = x_ + (K * z_diff);
  P_ = P_ - (K * S * K.transpose());

  /* Compute the NIS */
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
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
