#include "trajopt/belief.hpp"
#include <time.h>
#include <boost/bind.hpp>
#include "ipi/sco/expr_ops.hpp"
#include "ipi/sco/modeling_utils.hpp"
#include "utils/eigen_conversions.hpp"

using namespace std;
using namespace ipi::sco;
using namespace Eigen;
using namespace OpenRAVE;
using namespace util;

namespace trajopt {

BeliefRobotAndDOF::BeliefRobotAndDOF(OpenRAVE::RobotBasePtr _robot, const IntVec& _joint_inds, int _affinedofs, const OR::Vector _rotationaxis) :
			RobotAndDOF(_robot, _joint_inds, _affinedofs, _rotationaxis),
			generator(boost::mt19937(time(NULL)+rand()), boost::normal_distribution<>(0, 1))
{}

MatrixXd BeliefRobotAndDOF::GetDynNoise() {
	VectorXd diag_noise(3);
	diag_noise << 0.08, 0.13, 0.18;
	return diag_noise.asDiagonal();
}

MatrixXd BeliefRobotAndDOF::GetObsNoise() { return VectorXd::Constant(3,0.09).asDiagonal(); }

double sigmoid(double x,double mean) {
	double y = (x - mean);
	double s = (y/sqrt(1+y*y))+1.0;

	if (x < mean)
		return s*0.1;
	else
		return s*7.0;
}

VectorXd BeliefRobotAndDOF::Observe(const VectorXd& dofs, const VectorXd& r) {
	OR::RobotBasePtr robot = GetRobot();
	OR::RobotBase::RobotStateSaver saver = const_cast<BeliefRobotAndDOF*>(this)->Save();
	robot->SetDOFValues(toDblVec(dofs), false);
	OR::KinBody::LinkPtr link = robot->GetLink("Finger");
	OR::Vector trans = link->GetTransform().trans;

	VectorXd z = Vector3d(trans.x, trans.y, trans.z);
	//z += sigmoid(trans.y, -0.2)*GetObsNoise()*r;
	z += (0.5*pow(trans.y+0.2,2)+1)*GetObsNoise()*r; // as in the Platt paper
//	z += ((trans.y+0.2)/0.4)*GetObsNoise()*r;
//	if (trans.y<-0.2) z += 0.1*GetObsNoise()*r;
//	else z += 10*GetObsNoise()*r;
	return z;
}

VectorXd BeliefRobotAndDOF::Dynamics(const VectorXd& dofs, const VectorXd& u, const VectorXd& q) {
	VectorXd dofs1 = dofs+u + GetDynNoise()*q;
	return dofs1;
}

VectorXd BeliefRobotAndDOF::BeliefDynamics(const VectorXd& theta0, const VectorXd& u0) {
	VectorXd x0, x;
	MatrixXd rtSigma0, rtSigma;
	decomposeBelief(theta0, x0, rtSigma0);
	ekfUpdate(u0, x0, rtSigma0, x, rtSigma);
	VectorXd theta;
	composeBelief(x, rtSigma, theta);

//	cout << "rtSigma " << endl << rtSigma << endl;
//	cout << "theta " << theta.transpose() << endl << endl;
//	decomposeBelief(theta, x, rtSigma);
//	cout << "rtSigma " << endl << rtSigma << endl;
//	cout << "theta " << theta.transpose() << endl << endl;
//	cout << "---------------" << endl;

	return theta;
}

VectorXd BeliefRobotAndDOF::VectorXdRand(int size) {
	VectorXd v(size);
	for (int i=0; i<size; i++) v(i) = generator();
	return v;
}

void BeliefRobotAndDOF::composeBelief(const VectorXd& x, const MatrixXd& rt_S, VectorXd& theta) {
//	int n_dof = GetDOF();
//	theta.resize(n_dof + n_dof*(n_dof+1)/2);
//	theta.topRows(n_dof) = x;
//	int idx = n_dof;
//	for (int i=0; i<n_dof; i++) {
//		for (int j=i; j<n_dof; j++) {
//			theta(idx) = 0.5 * (rt_S(i,j)+rt_S(j,i));
//			idx++;
//		}
//	}
	int n_dof = GetDOF();
	theta.resize(n_dof + n_dof*(n_dof+1)/2);
	theta.topRows(n_dof) = x;
	int idx = n_dof;
	for (int j=0; j<n_dof; j++) {
		for (int i=j; i<n_dof; i++) {
			theta(idx) = rt_S(i,j);
			idx++;
		}
	}
}

void BeliefRobotAndDOF::decomposeBelief(const VectorXd& theta, VectorXd& x, MatrixXd& rt_S) {
//	int n_dof = GetDOF();
//	x = theta.topRows(n_dof);
//	int idx = n_dof;
//	rt_S.resize(n_dof, n_dof);
//	for (int j = 0; j < n_dof; ++j) {
//		for (int i = j; i < n_dof; ++i) {
//			rt_S(i,j) = rt_S(j,i) = theta(idx);
//			idx++;
//		}
//	}
	int n_dof = GetDOF();
	x = theta.topRows(n_dof);
	int idx = n_dof;
	rt_S = MatrixXd::Zero(n_dof, n_dof);
	for (int j=0; j<n_dof; j++) {
		for (int i=j; i<n_dof; i++) {
			rt_S(i,j) = theta(idx);
			idx++;
		}
	}
}

void BeliefRobotAndDOF::ekfUpdate(const VectorXd& u0, const VectorXd& x0, const MatrixXd& rtSigma0, VectorXd& x, MatrixXd& rtSigma) {
	int n_dof = GetDOF();

	VectorXd q = VectorXd::Zero(GetQSize());
	x = Dynamics(x0, u0, q);

	MatrixXd Sigma0 = rtSigma0 * rtSigma0.transpose();

	MatrixXd A = calcNumJac(boost::bind(&BeliefRobotAndDOF::Dynamics, this, _1, u0, q), x0);
	MatrixXd Gamma0 = A * Sigma0 * A.transpose();

	VectorXd r = VectorXd::Zero(GetRSize());
	MatrixXd C = calcNumJac(boost::bind(&BeliefRobotAndDOF::Observe, this, _1, r), x0);
	MatrixXd R = calcNumJac(boost::bind(&BeliefRobotAndDOF::Observe, this, x0, _1), r);

	MatrixXd W;
	MatrixXd A_K = C*Gamma0*C.transpose() + R*R.transpose();
	PartialPivLU<MatrixXd> solver(A_K);
	MatrixXd L = solver.solve(C*Gamma0);
	MatrixXd Sigma = Gamma0 - Gamma0 * C.transpose() * L;

	LLT<MatrixXd> lltofSigma(Sigma);
	rtSigma = lltofSigma.matrixL();
}

Eigen::MatrixXd BeliefRobotAndDOF::EndEffectorJacobian(const Eigen::VectorXd& x0) {
	Eigen::MatrixXd jac(3,3);
	double l1 = 0.16;
	double l2 = 0.16;
	double l3 = 0.08;
	double s1 = -l1 * sin(x0(0));
	double s2 = -l2 * sin(x0(0)+x0(1));
	double s3 = -l3 * sin(x0(0)+x0(1)+x0(2));
	double c1 = l1 * cos(x0(0));
	double c2 = l2 * cos(x0(0)+x0(1));
	double c3 = l3 * cos(x0(0)+x0(1)+x0(2));
	jac << s1+s2+s3, s2+s3, s3, c1+c2+c3, c2+c3, c3, 0, 0, 0;

//		// analytical jacobian computed in openrave
//		OR::KinBody::LinkPtr link = GetRobot()->GetLink("Finger");
//		DblMatrix Jxyz = PositionJacobian(link->GetIndex(), link->GetTransform().trans);
//		cout << "Jxyz" << endl;
//		std::cout << Jxyz << std::endl;
//		cout << "JAC" << endl;
//		cout << jac << endl;
//		cout << "-----------" << endl;

	return jac;
}


BeliefDynamicsConstraint::BeliefDynamicsConstraint(const VarVector& theta0_vars,	const VarVector& theta1_vars, const VarVector& u_vars, BeliefRobotAndDOFPtr brad) :
    		Constraint("BeliefDynamics"), brad_(brad), theta0_vars_(theta0_vars), theta1_vars_(theta1_vars), u_vars_(u_vars), type_(EQ)
{}

vector<double> BeliefDynamicsConstraint::value(const vector<double>& xin) {
	VectorXd theta0_hat = getVec(xin, theta0_vars_);
	VectorXd theta1_hat = getVec(xin, theta1_vars_);
	VectorXd u_hat = getVec(xin, u_vars_);

//	cout << "theta0_hat inside:\n" << theta0_hat.transpose() << endl;
//	cout << "theta1_hat inside:\n" << theta1_hat.transpose() << endl;
//	cout << "u_hat inside:\n" << u_hat.transpose() << endl;
	return toDblVec(brad_->BeliefDynamics(theta0_hat, u_hat) - theta1_hat);
}

///**
// *
// * y = f(x)
// * linearize around x0 to get
// * f(x) \approx y0 + grad f(x)|x0 *(x - x0)
// */
//AffExpr exprFromValGrad(double y0, const VectorXd& grad, const VarVector& vars, const VectorXd& x0) {
//	AffExpr y(y0);
//	exprInc(y, varDot(grad, vars));
//	exprDec(y, grad.dot(x0));
//	return y;
//}

ConvexConstraintsPtr BeliefDynamicsConstraint::convex(const vector<double>& xin, Model* model) {
	VectorXd theta0_hat = getVec(xin, theta0_vars_);
	VectorXd theta1_hat = getVec(xin, theta1_vars_);
	VectorXd u_hat = getVec(xin, u_vars_);

//	cout << "theta0_hat:\n" << theta0_hat.transpose() << endl;
//	cout << "theta1_hat:\n" << theta1_hat.transpose() << endl;
//	cout << "u_hat:\n" << u_hat.transpose() << endl;

	// linearize belief dynamics around theta0_hat and u_hat
	MatrixXd A = calcNumJac(boost::bind(&BeliefRobotAndDOF::BeliefDynamics, brad_.get(), _1, u_hat), theta0_hat);

	/*
	VectorXd theta_plus = theta0_hat, theta_minus = theta0_hat;
	double eps = 0.00048828125;
	theta_plus(1) += eps;
	theta_minus(1) -= eps;
	cout << (brad_.get()->BeliefDynamics(theta_plus, u_hat).transpose() - brad_.get()->BeliefDynamics(theta_minus, u_hat).transpose())/(2*eps) << endl;
	*/

	MatrixXd B = calcNumJac(boost::bind(&BeliefRobotAndDOF::BeliefDynamics, brad_.get(), theta0_hat, _1), u_hat);
	VectorXd c = brad_->BeliefDynamics(theta0_hat, u_hat);

//	cout << "A matrix:\n" << A << endl;
//	cout << "B matrix:\n" << B << endl;
//	cout << "c vector:\n" << c.transpose() << endl;

//	  // test convexification
//	  VectorXd theta0 = theta0_hat + brad_->VectorXdRand(9)*0.1;
//	  cout << "theta0" << theta0.transpose() << endl;
//	  VectorXd u = u_hat +brad_->VectorXdRand(3)*0.1;
//	  cout << "u "<< u.transpose() << endl;
//	  VectorXd diff1_approx = A * (theta0 - theta0_hat) + B * (u - u_hat) + c - theta1_hat;
//	  cout << "diff1_approx" << diff1_approx.transpose() << endl;
//		VectorXd diff1 = brad_->BeliefDynamics(theta0_hat, u_hat) - theta1_hat;
//	  cout << "diff1 " << diff1.transpose() << endl;

	// equality constraint
	// theta1_vars_ = A * (theta0_vars_ - theta0_hat) + B * (u_vars_ - u_hat) + c
	// 0 = A * (theta0_vars_ - theta0_hat) + B * (u_vars_ - u_hat) + c - theta1_vars_
	ConvexConstraintsPtr out(new ConvexConstraints(model));
	assert(A.rows() == B.rows());
	for (int i=0; i < A.rows(); ++i) {
		AffExpr aff_theta0;
		aff_theta0.constant = c[i] - A.row(i).dot(theta0_hat);
		aff_theta0.coeffs = toDblVec(A.row(i));
		aff_theta0.vars = theta0_vars_;
		AffExpr aff_u;
		aff_u.constant = - B.row(i).dot(u_hat);
		aff_u.coeffs = toDblVec(B.row(i));
		aff_u.vars = u_vars_;
		AffExpr aff_theta1;
		aff_theta1.constant = 0;
		aff_theta1.coeffs = vector<double>(theta1_vars_.size(),0);
		aff_theta1.coeffs[i] = 1;
		aff_theta1.vars = theta1_vars_;
		AffExpr aff_theta0_u = exprAdd(aff_theta0, aff_u);
		AffExpr aff = exprSub(aff_theta0_u, aff_theta1);
		aff = cleanupAff(aff);
		//cout << aff << "\n\n";

		//cout << aff.value(xin) << "\t" << value(xin)[i] << endl;

		out->addEqCnt(aff);
	}
	//cout << "-----------------" << endl;

	return out;
}

CovarianceCost::CovarianceCost(const VarVector& rtSigma_vars, const Eigen::MatrixXd& Q, BeliefRobotAndDOFPtr brad) :
    Cost("Covariance"), rtSigma_vars_(rtSigma_vars), Q_(Q), brad_(brad) {
	int n_dof = brad_->GetDOF();
	assert(rtSigma_vars_.size() == (n_dof*(n_dof+1)/2));
	assert(Q_.rows() == n_dof);
	assert(Q_.cols() == n_dof);

	QuadExpr a;
	a.coeffs = vector<double>(3,Q_(0,0));
	a.vars1.push_back(rtSigma_vars_[0]);
	a.vars1.push_back(rtSigma_vars_[1]);
	a.vars1.push_back(rtSigma_vars_[2]);
	a.vars2 = a.vars1;

	QuadExpr e;
	e.coeffs = vector<double>(2,Q_(1,1));
	e.vars1.push_back(rtSigma_vars_[3]);
	e.vars1.push_back(rtSigma_vars_[4]);
	e.vars2 = e.vars1;

	QuadExpr i;
	i.coeffs = vector<double>(1,Q_(2,2));
	i.vars1.push_back(rtSigma_vars_[5]);
	i.vars2 = i.vars1;

	QuadExpr bd;
	bd.coeffs = vector<double>(2,Q_(0,1)+Q_(1,0));
	bd.vars1.push_back(rtSigma_vars_[1]);
	bd.vars1.push_back(rtSigma_vars_[2]);
	bd.vars2.push_back(rtSigma_vars_[3]);
	bd.vars2.push_back(rtSigma_vars_[4]);

	QuadExpr cg;
	cg.coeffs = vector<double>(1,Q_(0,2)+Q_(2,0));
	cg.vars1.push_back(rtSigma_vars_[2]);
	cg.vars2.push_back(rtSigma_vars_[5]);

	QuadExpr fh;
	fh.coeffs = vector<double>(1,Q_(1,2)+Q_(2,1));
	fh.vars1.push_back(rtSigma_vars_[4]);
	fh.vars2.push_back(rtSigma_vars_[5]);

	exprInc(expr_,a);
	exprInc(expr_,e);
	exprInc(expr_,i);
	exprInc(expr_,bd);
	exprInc(expr_,cg);
	exprInc(expr_,fh);
	expr_ = cleanupQuad(expr_);
}

double CovarianceCost::value(const vector<double>& xin) {
	int n_dof = brad_->GetDOF();
	VectorXd rtSigma_vec = getVec(xin, rtSigma_vars_);
	MatrixXd rtSigma(n_dof, n_dof);
	VectorXd x_unused(n_dof);
	VectorXd theta(n_dof+rtSigma_vec.size());
	theta.bottomRows(rtSigma_vec.size()) = rtSigma_vec;
	brad_->decomposeBelief(theta, x_unused, rtSigma);

	return (Q_ * rtSigma.transpose() * rtSigma).trace();
}

ConvexObjectivePtr CovarianceCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}


MatrixXd calcNumJac(VectorOfVectorFun f, const VectorXd& x, double epsilon) {
	VectorXd y = f(x);
	MatrixXd out(y.size(), x.size());
	VectorXd x_plus = x;
	VectorXd x_minus = x;
	for (size_t i=0; i < size_t(x.size()); ++i) {
		x_plus(i) = x(i) + epsilon;
		VectorXd y_plus = f(x_plus);
		x_minus(i) = x(i) - epsilon;
		VectorXd y_minus = f(x_minus);
		out.col(i) = (y_plus - y_minus) / (2*epsilon);
		x_plus(i) = x(i);
		x_minus(i) = x(i);
	}
	return out;
}

osg::Matrix gaussianToTransform(const Eigen::Vector3d& mean, const Eigen::Matrix3d& cov) {
	//	Eigen::Matrix3d cov;
	//	cov << 0.025, 0.0075, 0.00175, 0.0075, 0.0070, 0.00135, 0.00175, 0.00135, 0.00043;
	//	Eigen::Vector3d mean(0,0,0.1);

	Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
	Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
	t.block(0,0,3,3) = es.eigenvectors().real() * es.eigenvalues().real().cwiseSqrt().asDiagonal();
	t.block(0,3,3,1) = mean;

	osg::Matrix osg_t;
	osg_t.set(t.data());
	return osg_t;
}

}
