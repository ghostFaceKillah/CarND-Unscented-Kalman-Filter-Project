#include <iostream>
#include "Eigen/Dense"

#include "ukf.h"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
  // use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // will be fully initialized with the first measurment
  is_initialized_ = false;

  // State size
  n_x_ = 5;

  // Augmented state size
  n_aug_ = n_x_ + 2;

  // scaling parameter for the unscented transform
  lambda_  = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // initialize predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.9;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // std_yawdd_ = 1.7; 1.3 was cool
  std_yawdd_ = 0.4;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // compute weights here for later reuse
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  // initialize radar noise matrix for reuse
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0,                         0,
              0,                     std_radphi_ * std_radphi_, 0,
              0,                     0,                         std_radrd_ * std_radrd_;

  // initlize Lidar Noise matrix for reuse
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0,                       std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

/**
 * Finish initialization of the Kalman filter object, using the first measurement.
 *
 * @param {MeasurementPackage} The first measurement processed by the system.
 */
void UKF::InitializeWithFirstMeasurement(MeasurementPackage meas_package) {
  x_.fill(0.0);

  float v_var;
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    x_(0) = meas_package.raw_measurements_(0);
    x_(1) = meas_package.raw_measurements_(1);
    v_var = 1;
  } else {
    float rho = meas_package.raw_measurements_(0);
    float phi = meas_package.raw_measurements_(1);
    float rho_dot = meas_package.raw_measurements_(2);

    x_(0) = rho * cos(phi);
    x_(1) = rho * sin(phi);
    x_(2) = rho_dot;

    v_var = 0.5;
  }
 
  P_ << 0.15,  0,    0,      0,    0,
        0,     0.15, 0,      0,    0, 
        0,     0,    1,      0,    0, 
        0,     0,    0,      1,    0, 
        0,     0,    0,      0,    1;

  time_us_ = meas_package.timestamp_;
  is_initialized_ = true;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {

    InitializeWithFirstMeasurement(meas_package);

  } else {

    double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
    time_us_ = meas_package.timestamp_;

    while (delta_t > 0.1) {
      const double dt = 0.05;
      Prediction(dt);
      delta_t -= dt;
    }

    Prediction(delta_t);

    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
      UpdateLidar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      UpdateRadar(meas_package);
    }
  }
}


/**
 * Convenience function, wrapping the sigma points calculation.
 */
MatrixXd UKF::GenerateSigmaPoints() {
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;


  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(n_aug_ + i + 1) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }


  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
      Xsig_aug(3, i) = Xsig_aug(3, i);
  };

  return Xsig_aug;
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    //
  // prepare augmented sigma points
  MatrixXd Xsig_aug = GenerateSigmaPoints();
  
  // predict sigma points
  for (int i = 0 ; i < 2 * n_aug_ + 1; ++i) {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);

    // state augmentation
    double nu_a = Xsig_aug(5, i); 
    double nu_yawdd = Xsig_aug(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 1e-3) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p += nu_a * delta_t;

    yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p += nu_yawdd * delta_t;

    // write the prediction to the output matrix column;
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }


  // calculate predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // calculate predicted state covariance
  P_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    x_diff(3) = normalize(x_diff(3));

    P_ += weights_(i) * (x_diff * x_diff.transpose());
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  int n_z = 2;

  VectorXd z = meas_package.raw_measurements_;

  // predicted measurements sigma points
  MatrixXd Zsig_pred(n_z, 2 * n_aug_ + 1);

  // predicted measurement mean
  VectorXd Z_pred(n_z);
  Z_pred.fill(0.0);

  // Predict sigma points measurements
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Zsig_pred(0, i) = Xsig_pred_(0, i);
    Zsig_pred(1, i) = Xsig_pred_(1, i);

    Z_pred += weights_(i) * Zsig_pred.col(i);
  }

  // Compute predicted measurement covariance
  MatrixXd S(n_z, n_z);

  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig_pred.col(i) - Z_pred;

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  S += R_lidar_;

  // calculate cross-correlation matrix
  MatrixXd Tc(n_x_, n_z);
  Tc.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd diffX = Xsig_pred_.col(i) - x_;
    VectorXd diffZ = Zsig_pred.col(i) - Z_pred;

    Tc += weights_(i) * diffX * diffZ.transpose();
  }

  // calculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  // Use Kalman gain to propagate innovation into state estimate
  VectorXd y = z - Z_pred;
  x_ += K * y;
  P_ -= K * S * K.transpose();

  NIS_laser_ = y.transpose() * S.inverse() * y;
}



/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // compute measurement prediction 
  MatrixXd Z_sig_pred(n_z, 2 * n_aug_ + 1);
  Z_sig_pred.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    Z_sig_pred(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Z_sig_pred(1, i) = atan2(p_y, p_x);
    Z_sig_pred(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
  }

  // Compute prediction mean
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++ i) {
    z_pred += weights_(i) * Z_sig_pred.col(i);
  }


  // Compute measurement prediction covariance
  MatrixXd S(n_z, n_z);

  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
      VectorXd diff = Z_sig_pred.col(i) - z_pred; ////////
      diff(1) = normalize(diff(1));
      S += weights_(i) * (diff * diff.transpose());
  }

  // add measurement noise
  S += R_radar_;

  // unpack incoming radar measurmenet
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;

  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    
    // residual
    VectorXd z_diff = Z_sig_pred.col(i) - z_pred; ///////////
    z_diff(1) = normalize(z_diff(1));

    // state difference 
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = normalize(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // residual 
  VectorXd z_diff = z - z_pred;
  z_diff(1) = normalize(z_diff(1));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
