#include "trajopt/plot_callback.hpp"
#include "trajopt/common.hpp"
#include "osgviewer/osgviewer.hpp"
#include "utils/eigen_conversions.hpp"
#include <boost/foreach.hpp>
#include "trajopt/problem_description.hpp"
using namespace OpenRAVE;
using namespace util;
using namespace Eigen;
using namespace std;
namespace trajopt {

void PlotTraj(OSGViewer& viewer, BeliefRobotAndDOFPtr rad, const TrajArray& traj, vector<GraphHandlePtr>& handles) {
	const int n_dof = rad->GetDOF();
	const int n_theta = rad->GetNTheta();
	const int n_steps = traj.rows();

	for (int i=0; i < n_steps; ++i) {
    rad->SetDOFValues(toDblVec(traj.block(i,0,1,n_dof).transpose()));
    handles.push_back(viewer.PlotKinBody(rad->GetRobot()));
		double trans_param = (((double)i)/((double)n_steps-1.0)+0.35)/1.35;
    SetTransparency(handles.back(), trans_param);
  }

	OR::RobotBase::RobotStateSaver saver = const_cast<BeliefRobotAndDOF*>(rad.get())->Save();

	for (int i=0; i < n_steps; ++i) {
		VectorXd theta = traj.block(i,0,1,n_theta).transpose();
		VectorXd mean;
		MatrixXd cov;
		rad->GetEndEffectorNoiseAsGaussian(theta, mean, cov);

		//handles.push_back(viewer.PlotEllipsoid(gaussianToTransform(trans_eig,cov), OR::Vector(1,0,0,1)));
    //SetTransparency(handles.back(), 0.35);
		double color_param = ((double)i)/((double)n_steps-1.0);
		handles.push_back(viewer.PlotEllipseXYContour(gaussianAsTransform(mean,cov), OR::Vector(0.0,color_param,1.0-color_param,1.0)));

	}

	VectorXd theta = traj.block(0,0,1,n_theta).transpose();
	for (int i=0; i<n_steps; i++) {
		VectorXd mean;
		MatrixXd cov;
		rad->GetEndEffectorNoiseAsGaussian(theta, mean, cov);

		double color_param = ((double)i)/((double)n_steps-1.0);
		handles.push_back(viewer.PlotEllipseXYContour(gaussianAsTransform(mean,cov), OR::Vector(0.0,color_param,1.0-color_param,1.0), true));

		if (i != (n_steps-1)) {
			VectorXd u = traj.block(i,n_theta,1,n_dof).transpose();
			theta = rad->BeliefDynamics(theta,u);
		}
	}
}

void PlotCosts(OSGViewer& viewer, vector<CostPtr>& costs, vector<ConstraintPtr>& cnts, BeliefRobotAndDOFPtr rad, const VarArray& vars, const DblVec& x) {
  vector<GraphHandlePtr> handles;
  handles.clear();
  BOOST_FOREACH(CostPtr& cost, costs) {
    if (Plotter* plotter = dynamic_cast<Plotter*>(cost.get())) {
      plotter->Plot(x, *rad->GetRobot()->GetEnv(), handles);
    }
  }
  BOOST_FOREACH(ConstraintPtr& cnt, cnts) {
    if (Plotter* plotter = dynamic_cast<Plotter*>(cnt.get())) {
      plotter->Plot(x, *rad->GetRobot()->GetEnv(), handles);
    }
  }
  TrajArray traj = getTraj(x, vars);
  PlotTraj(viewer, rad, traj, handles);
  viewer.Idle();
  rad->SetDOFValues(toDblVec(traj.row(traj.rows()-1)));
}



Optimizer::Callback PlotCallback(TrajOptProb& prob) {
  OSGViewerPtr viewer = OSGViewer::GetOrCreate(prob.GetEnv());
  vector<ConstraintPtr> cnts = prob.getConstraints();
  return boost::bind(&PlotCosts, boost::ref(*viewer),
                      boost::ref(prob.getCosts()),
                      cnts,
                      prob.GetRAD(),
                      boost::ref(prob.GetVars()),
                      _1);
}

}
