#include "trajopt/belief_constraints.hpp"
#include "trajopt/utils.hpp"
#include <boost/bind.hpp>
#include "sco/expr_ops.hpp"
#include "sco/expr_op_overloads.hpp"
#include "utils/eigen_conversions.hpp"

using namespace std;
using namespace sco;
using namespace Eigen;
using namespace OpenRAVE;
using namespace util;

namespace {
template <typename T>
vector<T> concat(const vector<T>& a, const vector<T>& b) {
	vector<T> out;
	vector<int> x;
	out.insert(out.end(), a.begin(), a.end());
	out.insert(out.end(), b.begin(), b.end());
	return out;
}

}
namespace trajopt {

BeliefDynamicsConstraint::BeliefDynamicsConstraint(const VarVector& theta0_vars,	const VarVector& theta1_vars, const VarVector& u_vars, BeliefRobotAndDOFPtr brad) :
    				Constraint("BeliefDynamics"), brad_(brad), theta0_vars_(theta0_vars), theta1_vars_(theta1_vars), u_vars_(u_vars), type_(EQ)
{}

vector<double> BeliefDynamicsConstraint::value(const vector<double>& xin) {
	VectorXd theta0_hat = getVec(xin, theta0_vars_);
	VectorXd theta1_hat = getVec(xin, theta1_vars_);
	VectorXd u_hat = getVec(xin, u_vars_);
	return toDblVec(brad_->BeliefDynamics(theta0_hat, u_hat) - theta1_hat);
}

/*
bool isMatrixNan(const MatrixXd& m) {
	for (int i=0; i<m.rows(); i++)
		for (int j=0; j<m.cols(); j++)
			if (isnan(m(i,j))) return true;
	return false;
}

bool assertMatrixNotNan(const MatrixXd& m) {
	for (int i=0; i<m.rows(); i++)
		for (int j=0; j<m.cols(); j++)
			assert(!isnan(m(i,j)));
}
*/

ConvexConstraintsPtr BeliefDynamicsConstraint::convex(const vector<double>& xin, Model* model) {
	VectorXd theta0_hat = getVec(xin, theta0_vars_);
	VectorXd theta1_hat = getVec(xin, theta1_vars_);
	VectorXd u_hat = getVec(xin, u_vars_);

	//	cout << "theta0_hat:\n" << theta0_hat.transpose() << endl;
	//	cout << "theta1_hat:\n" << theta1_hat.transpose() << endl;
	//	cout << "u_hat:\n" << u_hat.transpose() << endl;

	// linearize belief dynamics around theta0_hat and u_hat
	MatrixXd A = brad_->dgdb(theta0_hat, u_hat);
	VectorXd c = brad_->BeliefDynamics(theta0_hat, u_hat);
	MatrixXd B = brad_->dgdu(theta0_hat, u_hat);

	//	cout << "theta0_hat " << theta0_hat.transpose() << endl;
	//	cout << "u_hat " << u_hat.transpose() << endl;
	//	if (isMatrixNan(A)) cout << "A" << endl << A << endl;
	//	if (isMatrixNan(B)) cout << "B" << endl << B << endl;
	//	if (isMatrixNan(c)) cout << "c" << endl << c << endl;

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
		out->addEqCnt(aff);
	}

	return out;
}


struct BeliefDynamicsErrCalculator : public VectorOfVector {
	BeliefRobotAndDOFPtr brad_;
	BeliefDynamicsErrCalculator(BeliefRobotAndDOFPtr brad) : brad_(brad) {}

	VectorXd operator()(const VectorXd& vals) const {
		int b_dim = brad_->GetBDim();
		int u_dim = brad_->GetUDim();
		VectorXd theta0 = vals.topRows(b_dim);
		VectorXd theta1 = vals.middleRows(b_dim, b_dim);
		VectorXd u0 = vals.bottomRows(u_dim);
		assert(vals.size() == (2*b_dim + u_dim));

		VectorXd err = brad_->BeliefDynamics(theta0, u0) - theta1;

		return err;
	}
};

BeliefDynamicsConstraint2::BeliefDynamicsConstraint2(const VarVector& theta0_vars,	const VarVector& theta1_vars, const VarVector& u_vars,
		BeliefRobotAndDOFPtr brad, const BoolVec& enabled) :
		ConstraintFromFunc(VectorOfVectorPtr(new BeliefDynamicsErrCalculator(brad)),
				concat(concat(theta0_vars, theta1_vars), u_vars), EQ, "BeliefDynamics2")
{
}

CovarianceCost::CovarianceCost(const VarVector& rtSigma_vars, const MatrixXd& Q, BeliefRobotAndDOFPtr brad) :
    		Cost("Covariance"), rtSigma_vars_(rtSigma_vars), Q_(Q), brad_(brad)
{
	int s_dim = brad_->GetSDim();
	int q_dim = brad_->GetQDim();
	assert(rtSigma_vars_.size() == s_dim);
	assert(Q_.rows() == q_dim);

	VarArray rtSigma_matrix_vars = toBasicArray(brad_->toSigmaMatrix(toVectorXd(rtSigma_vars_)));
	QuadExprArray Sigma_vars = rtSigma_matrix_vars * rtSigma_matrix_vars.transpose();
	expr_ =  (Q_ * Sigma_vars).trace();
	expr_ = cleanupQuad(expr_);
}

double CovarianceCost::value(const vector<double>& xin) {
	int x_dim = brad_->GetXDim(), s_dim = brad_->GetSDim(), b_dim = brad_->GetBDim();
	VectorXd rtSigma_vec = getVec(xin, rtSigma_vars_);
	MatrixXd rtSigma(x_dim, x_dim);
	VectorXd x_unused(x_dim);
	VectorXd theta(b_dim);
	theta.bottomRows(s_dim) = rtSigma_vec;
	brad_->decomposeBelief(theta, x_unused, rtSigma);

	return (Q_ * rtSigma * rtSigma.transpose()).trace();
}

ConvexObjectivePtr CovarianceCost::convex(const vector<double>& x, Model* model) {
	ConvexObjectivePtr out(new ConvexObjective(model));
	out->addQuadExpr(expr_);
	return out;
}

ControlCost::ControlCost(const VarArray& vars, const VectorXd& coeffs) :
    		Cost("Control"), vars_(vars), coeffs_(coeffs) {
	for (int i=0; i < vars.rows(); ++i) {
		QuadExpr expr;
		expr.vars1 = vars_.row(i);
		expr.vars2 = vars_.row(i);
		expr.coeffs = toDblVec(coeffs);
		exprInc(expr_, expr);
	}
}
double ControlCost::value(const vector<double>& xvec) {
	MatrixXd traj = getTraj(xvec, vars_);
	return (traj.array().square().matrix() * coeffs_.asDiagonal()).sum();
}
ConvexObjectivePtr ControlCost::convex(const vector<double>& x, Model* model) {
	ConvexObjectivePtr out(new ConvexObjective(model));
	out->addQuadExpr(expr_);

	return out;
}

}
