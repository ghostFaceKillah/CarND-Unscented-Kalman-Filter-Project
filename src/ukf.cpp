#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Utils
double normalize(double x) { 
  while (x > M_PI) x -= 2.0 * M_PI;
  while (x(1) < -M_PI) x(1) += 2.0 * M_PI;
  return x;
};


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // will be fully initialized with the first measurment
  is_initlized_ = false;

  // State size
  n_x_ = 5;

  // Augmented state size
  n_aug_ = n_x_ + 2;

  // scaling parameter for the unscented transform
  lambda_  = 3 - n_aug;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // initialize predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // TODO(Mike): Tune
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // TODO(mike): Tune
  std_yawdd_ = 30;

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
  weights(0) = lambda / (lambda + n_aug);
  for (int i = 1; i < 2 * n_aug + 1; ++i) {
    weights(i) = 0.5 / (n_aug + lambda);
  }

  // initialize radar noise matrix for reuse
  R_radar = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0,                         0,
              0,                     std_radphi_ * std_radphi_, 0,
              0,                     0,                         std_radrd_ * std_radrd_;

  // initlize Lidar Noise matrix for reuse
  R_lidar = MatrixXd(2, 2);
  R_lidar_ << std_laspx * std_laspx, 0,
              0,                     std_laspy_ * std_laspy;
}

UKF::~UKF() {}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {
  VectorXd x_aug = VectorXd(7);
  MatrixXd P_aug = MatrixXd(7, 7);
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;


  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P;
  P_aug(5, 5) = std_a_ * std_a;
  P_aug(6, 6) = std_yawdd * std_yawdd;

  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug; ++i)
  {
    X_sig_aug.col(i + 1) = x_aug + sqrt(lambda + n_aug) * L.col(i);
    X_sig_aug.col(n_aug_ + i + 1) = x_aug - sqrt(lambda + n_aug) * L.col(i);
  }

  // TODO(mike): Perhaps normalize the angles here?

  *Xsig_out = Xsig_aug;
}

/**
 * Finish initialization
 *
 * @param {MeasurementPackage} The first measurement processed by the system.
 */
void UFK::InitializeWithFirstMeasurement(MeasurementPackage meas_package) {
  x_.setZero();

  float v_var;

  if (meas_package.sensor_type == MeasurementPackage::LASER) {
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

  P_ << 0.5,  0,    0,      0,    0,
        0,    0.5,  0,      0,    0, 
        0,    0,    v_var,  0,    0, 
        0,    0,    0,      0.5,  0, 
        0,    0,    0,      0,    0.5;

  time_us_ = meas_package.timestamp_;
  is_initialized_ = true;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_) {
    InitializeWithFirstMeasurement(meas_package);
  } else {

    double delta_t = (meas_package.timestamp_ - time_us) / 1e6;
    time_us = meas_package.timestamp_;

    // while (delta_t > 0.1) {
    //   const double dt = 0.05;
    //   Prediction(dt);
    //   delta_t -= dt;
    // }

    Prediction(delta_t);

    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
      UpdateLidar(meas_package);
    } else {
      UpdateRadar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // prepare augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
  GenerateSigmaPoints(&Xsig_out);
  
  // predict sigma points
  for (int i = 0 ; i < 2 * n_aug + 1; ++i) {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    // state augmentation
    double nu_a = Xsig_aug(5, i);
    double nu_yawwed = Xsig_aug(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 1e-3) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw + yawd * delta_t) - cos(yaw));
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
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }


  // TODO: Mike - Break it up as another function
  // calculate predicted mean and covariance

  // create vector for predicted new state
  VectorXd x = Vector(n_x);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x, n_x);

  // calculate predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; ++i) {
    x += weights_(i) * Xsig_pred.col(i);
  }

  // calculate predicted state covariance
  P.fill(0.0);

  for (int i = 0; i < 2 * n_aug + 1; ++i) {
    // innovation
    VectorXd x_diff = Xsig_pred.col(i) - x;

    // angle normalization
    x_diff(3) = normalize(x_diff(3)):

    P += weights_(i) * x_diff * x_diff.transpose();
  }

  x_ = x;
  P_ = P;
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

  int n_z = 2;
  VectorXd z(n_z);
  z = meas_package.raw_measurments_;

  // predicted measurements sigma points
  MatrixXd Zsig_pred(n_z, 2 * n_aug_ + 1);

  // predicted measurement mean
  VectorXd Z_pred(n_z);
  Z_pred.fill(0.0);

  // Predict sigma points measurements
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Zsig_pred(0, i) = Xsig_pred(0, i);
    Zsig_pred(1, i) = Xsig_pred(1, i);

    Z_pred += weights_(i) * Zsig_pred.col(i);
  }

  // Compute predicted measurement covariance
  MatrixXd S(n_z, n_z);

  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd diffVec = Zsig_pred.col(i) - Z_pred;

    S += weights_(i) * (diffVec * diffVec.transpose());
  }

  S += R_lidar_;

  // calculate cross-correlation matrix
  MatrixXd Tc(n_x_, n_z);
  Tc.setZero();

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd diffX = Xsig_pred.col(i) - x_;
    VectorXd diffZ = Zsig_pred.col(i) - Z_pred;

    Tc += weights_(i) * diffX * diffZ.transpose();
  }

  MatrixXd K = Tc * S.inverse();

  VectorXd y = z - Z_pred;
  X_ += K * y;
  P_ -= K * S * K.transpose();

  NIS_laser_ = y.transpose() * Si * y
}

VectorXd UKF::PredictRadarMeasurement() {
  int n_z = 3;

  MatrixXd Z_sig(n_z, 2 * n_aug + 1);

  for (int i = 0; i < 2 * n_aug + 1; ++i) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred(2, i);
    double yaw = Xsig_pred(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // r
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    // phi
    Zsig(1, i) = atan2(p_y, p_x);
    // r_dot
    Zsig(2, i) = (p_x * v1, p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
  }

  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; ++ i) {
    z_pred += weights(i) * Zsig.coli(i);
  }

  return z_pred;
}


/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // compute measurement prediction 
  VectorXd z_pred(n_z);
  z_pred = PredictRadarMeasurement();

  // unpack incoming radar  measurmenet
  VectorXd z = Vector(n_z);
  z = meas_package_.raw_measurements_;
  z(1) = normalize(z(1));

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; ++i) {
    
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = normalize(z_diff(1));

    // state difference 
    VectorXd x_diff = Xsig_pred.col(i) - x;
    x_diff(3) = normalize(x_diff(3)):

    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // residual 
  VectorXd z_diff = z - z_pred;
  z_diff(1) = normalize(z_diff(1));

  x += K * z_diff;
  P += P - K * S * K.transpose();
}
