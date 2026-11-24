#ifndef PALM_KINEMATICS_HPP
#define PALM_KINEMATICS_HPP

#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <string>
#include <vector>

// =============================================================================
// PalmKinematics: where are the palms and feet, RELATIVE TO THE PELVIS?
//
// WHY:   the policy needs to know how far each palm is from its target. On the real
//        robot we measure joint angles (encoders) and the pelvis tilt (IMU), but
//        nothing measures the palm directly. So we compute the palm position from
//        the joint angles: FORWARD KINEMATICS (FK).
//
// HOW:   MuJoCo does the math. We load the same robot XML the policy trained on, so
//        the palm and foot points ("sites") are exactly the ones training used.
//
// FRAME: all results are in the pelvis's own axes: x forward, y left, z up. The
//        pelvis tilt (roll, pitch) and its height above the floor are NOT included
//        here; the hand_pose_error observation adds them (IMU + lower foot).
//
// PIPELINE of compute():
//
//   joint angles         data->qpos          mj_kinematics()       palmPos, palmRot,
//   27, policy order --> MuJoCo's state --> every body & site --> footPos
//                        (one flat array)    position/rotation     (pelvis frame)
//
// USAGE:
//   PalmKinematics fk(xmlPath, jointNames);  // once, at startup
//   fk.compute(jointPos);                    // every policy step (microseconds)
//   fk.palmPos[0];                           // left palm position [m]
// =============================================================================
class PalmKinematics {
private:
    // --- MuJoCo objects ---
    mjModel* model;  // the robot DESCRIPTION: link lengths, joint axes, site offsets.
                     // Fixed after loading (like a URDF).
    mjData* data;    // the robot STATE: joint angles (qpos) and everything MuJoCo
                     // computes from them (site positions, ...). Changes every step.

    // --- Lookup tables, filled once in the constructor ---
    std::vector<int> qposIndex;  // qposIndex[i] = where policy joint i's angle goes in
                                 // data->qpos (found by joint NAME, so the joint order
                                 // in the XML doesn't matter).
    int palmSite[2];             // MuJoCo's id number of each palm site. 0 = left, 1 = right.
    int footSite[2];             // MuJoCo's id number of each foot site. 0 = left, 1 = right.

    // --- Small helpers ---
    int findSite(std::string name);          // site name -> id; stops the program if missing
    Eigen::Vector3d sitePosition(int site);  // read one site's position [m]
    Eigen::Matrix3d siteRotation(int site);  // read one site's 3x3 rotation matrix

public:
    // --- Results of the last compute(), in the pelvis frame. 0 = left, 1 = right. ---
    Eigen::Vector3d palmPos[2];  // palm position [m]
    Eigen::Matrix3d palmRot[2];  // palm orientation: column 0/1/2 = the palm's own
                                 // x/y/z axis, written in pelvis coordinates
    Eigen::Vector3d footPos[2];  // foot site position [m]; used later to estimate the
                                 // pelvis height (the lower foot stands on the floor)

    // Load the XML and look up every joint and site. Call ONCE at startup (~0.1 s).
    // A wrong path or name stops the program here, before the robot moves.
    PalmKinematics(std::string xmlPath, std::vector<std::string> jointNames);

    // Free MuJoCo's memory. MuJoCo allocated it, so MuJoCo must free it.
    ~PalmKinematics();

    // Run FK for these joint angles [rad], 27 numbers in the same order as jointNames
    // (POLICY order), and fill palmPos, palmRot and footPos.
    void compute(const Eigen::VectorXf& jointPos);
};

#endif
