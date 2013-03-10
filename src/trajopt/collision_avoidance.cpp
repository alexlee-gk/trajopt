#include "trajopt/collision_avoidance.hpp"
#include "trajopt/rave_utils.hpp"
#include "trajopt/utils.hpp"
#include "sco/expr_vec_ops.hpp"
#include "sco/expr_ops.hpp"
#include "sco/sco_common.hpp"
#include <boost/foreach.hpp>
#include "utils/eigen_conversions.hpp"
#include "sco/modeling_utils.hpp"
#include "utils/stl_to_string.hpp"
using namespace OpenRAVE;
using namespace sco;
using namespace util;
using namespace std;

namespace trajopt {


void CollisionsToDistances(const vector<Collision>& collisions, const Link2Int& m_link2ind,
    DblVec& dists, DblVec& weights) {
  // Note: this checking (that the links are in the list we care about) is probably unnecessary
  // since we're using LinksVsAll
  dists.clear();
  weights.clear();
  dists.reserve(collisions.size());
  weights.reserve(collisions.size());
  BOOST_FOREACH(const Collision& col, collisions) {
    Link2Int::const_iterator itA = m_link2ind.find(col.linkA);
    Link2Int::const_iterator itB = m_link2ind.find(col.linkB);
    if (itA != m_link2ind.end() || itB != m_link2ind.end()) {
      dists.push_back(col.distance);
      weights.push_back(col.weight);
    }
  }
}

void CollisionsToDistanceExpressions(const vector<Collision>& collisions, RobotAndDOF& rad,
    const Link2Int& link2ind, const VarVector& vars, const DblVec& dofvals, vector<AffExpr>& exprs, DblVec& weights) {

  exprs.clear();
  weights.clear();
  exprs.reserve(collisions.size());
  weights.reserve(collisions.size());
  rad.SetDOFValues(dofvals); // since we'll be calculating jacobians
  BOOST_FOREACH(const Collision& col, collisions) {
    AffExpr dist(col.distance);
    Link2Int::const_iterator itA = link2ind.find(col.linkA);
    if (itA != link2ind.end()) {
      VectorXd dist_grad = toVector3d(col.normalB2A).transpose()*rad.PositionJacobian(itA->second, col.ptA);
      exprInc(dist, varDot(dist_grad, vars));
      exprInc(dist, -dist_grad.dot(toVectorXd(dofvals)));
    }
    Link2Int::const_iterator itB = link2ind.find(col.linkB);
    if (itB != link2ind.end()) {
      VectorXd dist_grad = -toVector3d(col.normalB2A).transpose()*rad.PositionJacobian(itB->second, col.ptB);
      exprInc(dist, varDot(dist_grad, vars));
      exprInc(dist, -dist_grad.dot(toVectorXd(dofvals)));
    }
    if (itA != link2ind.end() || itB != link2ind.end()) {
      exprs.push_back(dist);
      weights.push_back(col.weight);
    }
  }
  RAVELOG_DEBUG("%i distance expressions\n", exprs.size());
}

void CollisionsToDistanceExpressions(const vector<Collision>& collisions, RobotAndDOF& rad, const Link2Int& link2ind,
    const VarVector& vars0, const VarVector& vars1, const DblVec& vals0, const DblVec& vals1,
    vector<AffExpr>& exprs, DblVec& weights) {
  vector<AffExpr> exprs0, exprs1;
  DblVec weights0, weights1;
  CollisionsToDistanceExpressions(collisions, rad, link2ind, vars0, vals0, exprs0, weights0);
  CollisionsToDistanceExpressions(collisions, rad, link2ind, vars1, vals1, exprs1, weights1);

  exprs.resize(exprs0.size());
  weights.resize(exprs0.size());

  for (int i=0; i < exprs0.size(); ++i) {
    exprScale(exprs0[i], (1-collisions[i].time));
    exprScale(exprs1[i], collisions[i].time);
    exprs[i] = AffExpr(0);
    exprInc(exprs[i], exprs0[i]);
    exprInc(exprs[i], exprs1[i]);
    weights[i] = (weights0[i] + weights1[i])/2;
  }
}

void BeliefCollisionsToDistanceExpressions(const vector<Collision>& collisions, BeliefRobotAndDOF& brad,
    const Link2Int& link2ind, const VarVector& theta_vars, const DblVec& theta_vals, vector<AffExpr>& exprs, DblVec& weights) {
  exprs.clear();
  weights.clear();
  exprs.reserve(collisions.size());
  weights.reserve(collisions.size());
  brad.SetBeliefValues(theta_vals); // since we'll be calculating jacobians
  BOOST_FOREACH(const Collision& col, collisions) {
  	Link2Int::const_iterator itA = link2ind.find(col.linkA);
		Link2Int::const_iterator itB = link2ind.find(col.linkB);
		AffExpr dist;
		for (int i=0; i<col.mi.alpha.size(); i++) {
			//cout << "ALPHA " << col.mi.alpha[i] << endl;
  		AffExpr dist_a(col.distance);
			if (itA != link2ind.end()) {
				VectorXd dist_grad = toVector3d(col.normalB2A).transpose()*brad.BeliefJacobian(itA->second, col.mi.instance_ind[i], col.ptA);
				exprInc(dist_a, varDot(dist_grad, theta_vars));
				exprInc(dist_a, -dist_grad.dot(toVectorXd(theta_vals)));
			}
			if (itB != link2ind.end()) {
				VectorXd dist_grad = -toVector3d(col.normalB2A).transpose()*brad.BeliefJacobian(itB->second, col.mi.instance_ind[i], col.ptB);
				exprInc(dist_a, varDot(dist_grad, theta_vars));
				exprInc(dist_a, -dist_grad.dot(toVectorXd(theta_vals)));
			}
			if (itA != link2ind.end() || itB != link2ind.end()) {
		    exprScale(dist_a, col.mi.alpha[i]);
		    exprInc(dist, dist_a);
			}
		}
		if (dist.constant!=0 || dist.coeffs.size()!=0 || dist.vars.size()!=0) {
			exprs.push_back(dist);
			weights.push_back(col.weight);
		}
  }
  RAVELOG_DEBUG("%i distance expressions\n", exprs.size());
}

void CollisionEvaluator::GetCollisionsCached(const DblVec& x, vector<Collision>& collisions) {
  double key = vecSum(x);
  vector<Collision>* it = m_cache.get(key);
  if (it != NULL) {
    RAVELOG_DEBUG("using cached collision check\n");
    collisions = *it;
  }
  else {
    RAVELOG_DEBUG("not using cached collision check\n");
    CalcCollisions(x, collisions);
    m_cache.put(key, collisions);
  }
}

SingleTimestepCollisionEvaluator::SingleTimestepCollisionEvaluator(RobotAndDOFPtr rad, const VarVector& vars) :
  m_env(rad->GetRobot()->GetEnv()),
  m_cc(CollisionChecker::GetOrCreate(*m_env)),
  m_rad(rad),
  m_vars(vars),
  m_link2ind(),
  m_links() {
  RobotBasePtr robot = rad->GetRobot();
  const vector<KinBody::LinkPtr>& robot_links = robot->GetLinks();
  vector<KinBody::LinkPtr> links;
  vector<int> inds;
  rad->GetAffectedLinks(m_links, true, inds);
  for (int i=0; i < m_links.size(); ++i) {
    m_link2ind[m_links[i].get()] = inds[i];
  }
}


void SingleTimestepCollisionEvaluator::CalcCollisions(const DblVec& x, vector<Collision>& collisions) {
  DblVec dofvals = getDblVec(x, m_vars);
  m_rad->SetDOFValues(dofvals);
  m_cc->LinksVsAll(m_links, collisions);
}

void SingleTimestepCollisionEvaluator::CalcDists(const DblVec& x, DblVec& dists, DblVec& weights) {
  vector<Collision> collisions;
  GetCollisionsCached(x, collisions);
  CollisionsToDistances(collisions, m_link2ind, dists, weights);
}


void SingleTimestepCollisionEvaluator::CalcDistExpressions(const DblVec& x, vector<AffExpr>& exprs, DblVec& weights) {
  vector<Collision> collisions;
  GetCollisionsCached(x, collisions);
  DblVec dofvals = getDblVec(x, m_vars);
  CollisionsToDistanceExpressions(collisions, *m_rad, m_link2ind, m_vars, dofvals, exprs, weights);
}

////////////////////////////////////////

CastCollisionEvaluator::CastCollisionEvaluator(RobotAndDOFPtr rad, const VarVector& vars0, const VarVector& vars1) :
  m_env(rad->GetRobot()->GetEnv()),
  m_cc(CollisionChecker::GetOrCreate(*m_env)),
  m_rad(rad),
  m_vars0(vars0),
  m_vars1(vars1),
  m_link2ind(),
  m_links() {
  RobotBasePtr robot = rad->GetRobot();
  const vector<KinBody::LinkPtr>& robot_links = robot->GetLinks();
  vector<KinBody::LinkPtr> links;
  vector<int> inds;
  rad->GetAffectedLinks(m_links, true, inds);
  for (int i=0; i < m_links.size(); ++i) {
    m_link2ind[m_links[i].get()] = inds[i];
  }
}

void CastCollisionEvaluator::CalcCollisions(const DblVec& x, vector<Collision>& collisions) {
  DblVec dofvals0 = getDblVec(x, m_vars0);
  DblVec dofvals1 = getDblVec(x, m_vars1);
  m_rad->SetDOFValues(dofvals0);
  m_cc->CastVsAll(*m_rad, m_links, dofvals0, dofvals1, collisions);
}
void CastCollisionEvaluator::CalcDistExpressions(const DblVec& x, vector<AffExpr>& exprs, DblVec& weights) {
  vector<Collision> collisions;
  GetCollisionsCached(x, collisions);
  DblVec dofvals0 = getDblVec(x, m_vars0);
  DblVec dofvals1 = getDblVec(x, m_vars1);
  CollisionsToDistanceExpressions(collisions, *m_rad, m_link2ind, m_vars0, m_vars1, dofvals0, dofvals1, exprs, weights);
}
void CastCollisionEvaluator::CalcDists(const DblVec& x, DblVec& dists, DblVec& weights) {
  vector<Collision> collisions;
  GetCollisionsCached(x, collisions);
  CollisionsToDistances(collisions, m_link2ind, dists, weights);
}


//////////////////////////////////////////
SigmaPtsCollisionEvaluator::SigmaPtsCollisionEvaluator(BeliefRobotAndDOFPtr rad, const VarVector& theta_vars) :
  m_env(rad->GetRobot()->GetEnv()),
  m_cc(CollisionChecker::GetOrCreate(*m_env)),
  m_rad(rad),
  m_theta_vars(theta_vars),
  m_link2ind(),
  m_links() {
  RobotBasePtr robot = rad->GetRobot();
  const vector<KinBody::LinkPtr>& robot_links = robot->GetLinks();
  vector<KinBody::LinkPtr> links;
  vector<int> inds;
  rad->GetAffectedLinks(m_links, true, inds);
  for (int i=0; i < m_links.size(); ++i) {
    m_link2ind[m_links[i].get()] = inds[i];
  }
}
void SigmaPtsCollisionEvaluator::CalcCollisions(const DblVec& x, vector<Collision>& collisions) {
  DblVec theta = getDblVec(x, m_theta_vars);
  MatrixXd sigma_pts = m_rad->sigmaPoints(toVectorXd(theta));

  vector<DblVec> dofvals(sigma_pts.cols());
  for (int i=0; i<sigma_pts.cols(); i++) {
  	dofvals[i] = toDblVec(sigma_pts.col(i));
  }
  m_cc->MultiCastVsAll(*m_rad, m_links, dofvals, collisions);
}

void SigmaPtsCollisionEvaluator::CustomPlot(const DblVec& x, std::vector<OR::GraphHandlePtr>& handles) {
	DblVec theta = getDblVec(x, m_theta_vars);
	MatrixXd sigma_pts = m_rad->sigmaPoints(toVectorXd(theta));
	vector<DblVec> dofvals(sigma_pts.cols());
	for (int i=0; i<sigma_pts.cols(); i++) {
		dofvals[i] = toDblVec(sigma_pts.col(i));
	}
	m_cc->PlotCastHull(*m_rad, m_links, dofvals, handles);
}

void SigmaPtsCollisionEvaluator::CalcDistExpressions(const DblVec& x, vector<AffExpr>& exprs, DblVec& weights) {
  vector<Collision> collisions;
  GetCollisionsCached(x, collisions);
  DblVec theta = getDblVec(x, m_theta_vars);
  // MatrixXd sigma_pts = m_rad->sigmaPoints(toVectorXd(theta));
  // is sigma_pts not being used ??
  BeliefCollisionsToDistanceExpressions(collisions, *m_rad, m_link2ind, m_theta_vars, theta, exprs, weights);
}
void SigmaPtsCollisionEvaluator::CalcDists(const DblVec& x, DblVec& dists, DblVec& weights) {
  vector<Collision> collisions;
  GetCollisionsCached(x, collisions);
	CollisionsToDistances(collisions, m_link2ind, dists, weights);
}
//////////////////////////////////////////


typedef OpenRAVE::RaveVector<float> RaveVectorf;

void PlotCollisions(const std::vector<Collision>& collisions, OR::EnvironmentBase& env, vector<OR::GraphHandlePtr>& handles, double safe_dist) {
  BOOST_FOREACH(const Collision& col, collisions) {
    RaveVectorf color;
    if (col.distance < 0) color = RaveVectorf(1,0,0,1);
    else if (col.distance < safe_dist) color = RaveVectorf(1,1,0,1);
    else color = RaveVectorf(0,1,0,1);
    handles.push_back(env.drawarrow(col.ptA, col.ptB, .0025, color));
  }
}

CollisionCost::CollisionCost(double dist_pen, double coeff, RobotAndDOFPtr rad, const VarVector& vars) :
    Cost("collision"),
    m_dist_pen(dist_pen),
    m_coeff(coeff)
{
	if (vars.size() == rad->GetDOF()) {
		m_calc = CollisionEvaluatorPtr(new SingleTimestepCollisionEvaluator(rad, vars));
	} else {
		BeliefRobotAndDOFPtr brad = boost::static_pointer_cast<BeliefRobotAndDOF>(rad);
		m_calc = CollisionEvaluatorPtr(new SigmaPtsCollisionEvaluator(brad, vars));
	}
}

CollisionCost::CollisionCost(double dist_pen, double coeff, RobotAndDOFPtr rad, const VarVector& vars0, const VarVector& vars1) :
    Cost("cast_collision"),
    m_calc(new CastCollisionEvaluator(rad, vars0, vars1)), m_dist_pen(dist_pen), m_coeff(coeff)
{}

ConvexObjectivePtr CollisionCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  vector<AffExpr> exprs;
  DblVec weights;
  m_calc->CalcDistExpressions(x, exprs, weights);
  for (int i=0; i < exprs.size(); ++i) {
    AffExpr viol = exprSub(AffExpr(m_dist_pen), exprs[i]);
    out->addHinge(viol, m_coeff*weights[i]);
  }
  return out;
}
double CollisionCost::value(const vector<double>& x) {
  DblVec dists, weights;
  m_calc->CalcDists(x, dists, weights);
  double out = 0;
  for (int i=0; i < dists.size(); ++i) {
    out += pospart(m_dist_pen - dists[i]) * m_coeff * weights[i];
  }
  return out;
}

void CollisionCost::Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles) {
  vector<Collision> collisions;
  m_calc->GetCollisionsCached(x, collisions);
  PlotCollisions(collisions, env, handles, m_dist_pen);
  m_calc->CustomPlot(x, handles);
}



}
