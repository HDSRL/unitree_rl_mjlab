// Step-1 check: forward kinematics at the default standing pose must give the
// hand library's default palms. Run it from deploy/robots/g1/build:
//   ./test_palm_kinematics
#include "palm_kinematics.hpp"
#include <iostream>

int main()
{
    std::string xml = "../../../../src/assets/robots/unitree_g1/xmls/g1_27dof.xml";

    // The policy's 27 joints, in POLICY order.
    std::vector<std::string> joints = {
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"};

    PalmKinematics fk(xml, joints);

    // Default standing pose, policy order (deploy.yaml default_joint_pos).
    Eigen::VectorXf q(27);
    q << -0.1, 0, 0, 0.3, -0.2, 0,  -0.1, 0, 0, 0.3, -0.2, 0,  0,
         0.35, 0.18, 0, 0.87, 0, 0, 0,  0.35, -0.18, 0, 0.87, 0, 0, 0;
    fk.compute(q);

    // The library put the pelvis 0.79 m above the floor, so add 0.79 to z.
    Eigen::Vector3d up(0.0, 0.0, 0.79);
    std::cout << "left  palm: " << (fk.palmPos[0] + up).transpose() << "   expected 0.0288  0.2288 0.6596\n";
    std::cout << "right palm: " << (fk.palmPos[1] + up).transpose() << "   expected 0.0288 -0.2288 0.6596\n";
    std::cout << "left palm rotation:\n" << fk.palmRot[0] << "\n";
    std::cout << "expected:\n 0.3449 -0.0338  0.9380\n 0.1851  0.9822 -0.0327\n-0.9202  0.1849  0.3450\n";
    std::cout << "feet z below pelvis: " << fk.footPos[0].z() << " " << fk.footPos[1].z() << "   expected -0.7842\n";
    return 0;
}
