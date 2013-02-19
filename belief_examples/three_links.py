import trajoptpy
import openravepy as rave
import numpy as np
import json
import trajoptpy.math_utils as mu
import trajoptpy.kin_utils as ku
import trajoptpy.make_kinbodies as mk

def move_arm_to_grasp(xyz_targ, quat_targ, link_name, manip_name):
    
    request = {
        "basic_info" : {
            "n_steps" : 20,
            "manip" : manip_name,
            "start_fixed" : True
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [1]}
        },        
        ],
        "constraints" : [
        {
            "type" : "pose",
            "name" : "final_pose",
            "params" : {
                "pos_coeffs" : [1,1,1],
                "rot_coeffs" : [1,1,1],
                "xyz" : list(xyz_targ),
                "wxyz" : list(quat_targ),
                "link" : link_name,
            },
        },
        ],
        "init_info" : {
            "type" : "stationary"
        }
    }
    
    return request



if __name__ == "__main__":
        
    ### Parameters ###
    ENV_FILE = "../data/three_links.env.xml"
    MANIP_NAME = "arm"
    LINK_NAME = "Finger"
    INTERACTIVE = True
    ##################
    
    ### Env setup ####
    env = rave.RaveGetEnvironment(1)
    if env is None:
        env = rave.Environment()
        env.StopSimulation()
        env.Load(ENV_FILE)
    robot = env.GetRobots()[0]
    robot.SetDOFValues([ 0 ,0 ,0]);
    manip = robot.GetManipulator(MANIP_NAME)
    ##################
    T_gripper = robot.GetLink(LINK_NAME).GetTransform()
    T_grasp = T_gripper.copy()
    T_grasp[:3,3]  += np.array([-0.5,0.2,0])
    T_grasp = T_grasp.dot(rave.matrixFromAxisAngle([0,0,1],np.pi/2))
    xyz_targ = T_grasp[:3,3]
    #success = mk.create_cylinder(env, xyz_targ-np.array([.03,0,0]), .02, .5)
    quat_targ = rave.quatFromRotationMatrix(T_grasp[:3,:3])

    request = move_arm_to_grasp(xyz_targ, quat_targ, LINK_NAME, MANIP_NAME)
    s = json.dumps(request)
    print "REQUEST:",s
    trajoptpy.SetInteractive(INTERACTIVE);
    prob = trajoptpy.ConstructProblem(s, env)
    result = trajoptpy.OptimizeProblem(prob)