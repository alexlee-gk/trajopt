#pragma once
#include <Eigen/Dense>
using namespace Eigen;


Vector4d quatMult(const Vector4d& q1, const Vector4d& q2) {
  return Vector4d(
      q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
      q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
      q1[0] * q2[2] + q1[2] * q2[0] + q1[3] * q2[1] - q1[1] * q2[3],
      q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]);
}

Vector4d quatExp(const Vector3d& r) {
  // see http://www.lce.hut.fi/~ssarkka/pub/quat.pdf
  double normr = r.norm();
  if (normr > 1e-10) {
    Vector4d q;
    q(0) = cos(normr / 2);
    q.bottomRows(3) = (r/normr) * sin(normr/2);
    return q;
  }
  else return Vector4d(1,0,0,0);
}

Vector3d quatLog(const Vector4d& q) {
  Vector3d v = q.bottomRows(3);
  double s = q(0);
  Vector3d out = (acos(s) / v.norm()) * v;
  return out;
}
Vector4d quatInv(const Vector4d& q) {
  Vector4d qinv = q;
  qinv.bottomRows(3) *= -1;
  return qinv;
}

MatrixXd getW(const MatrixXd& qs, double dt) {
  MatrixXd out(qs.rows()-1, 3);
  for (int i=0; i < out.rows(); ++i) {
    out.row(i) = (2/dt)*quatLog(quatMult(quatInv(qs.row(i)), (qs.row(i+1))));
  }
  // cout << "qs: " << endl << qs << endl;
  return out;
}
