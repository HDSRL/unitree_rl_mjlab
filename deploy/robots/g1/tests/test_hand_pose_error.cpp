// Step-3 check of hand_pose_error on a known robot state. Run it from build/:
//   ./test_hand_pose_error
#include "hand_observations.hpp"
#include <cmath>
#include <iostream>

using namespace isaaclab::mdp;  // so we can write computeHandPoseError(...) without a prefix

// Print the 24 numbers, 12 per hand: position error, tanh of it, rotation 6D.
void show(const std::string& title, const std::vector<float>& e)
{
    std::cout << title << "\n";
    for (int h = 0; h < 2; h++) {
        std::cout << (h == 0 ? "  left  " : "  right ");
        std::cout << "pos:";
        for (int i = 0; i < 3; i++) std::cout << " " << e[12 * h + i];
        std::cout << " | tanh:";
        for (int i = 3; i < 6; i++) std::cout << " " << e[12 * h + i];
        std::cout << " | 6D:";
        for (int i = 6; i < 12; i++) std::cout << " " << e[12 * h + i];
        std::cout << "\n";
    }
}

int main()
{
    PalmKinematics fk("../../../../src/assets/robots/unitree_g1/xmls/g1_27dof.xml", policyJointNames());

    // Default standing pose, policy order (deploy.yaml default_joint_pos).
    Eigen::VectorXf q(27);
    q << -0.1, 0, 0, 0.3, -0.2, 0,  -0.1, 0, 0, 0.3, -0.2, 0,  0,
         0.35, 0.18, 0, 0.87, 0, 0, 0,  0.35, -0.18, 0, 0.87, 0, 0, 0;
    fk.compute(q);

    // Targets = the default palms, so the errors should be near zero.
    std::vector<float> target = {0.0288f, 0.22884f, 0.65959f, 0.0288f, -0.22884f, 0.65959f};
    std::vector<float> targetQuat = {0.81733f, 0.06654f, 0.56838f, 0.06694f, 0.81733f, -0.06654f, 0.56838f, -0.06694f};

    // Case 1: pelvis level (IMU identity). Expected: z error about +0.006 m (the
    // library assumed a pelvis height of 0.79 m; the feet say 0.784 m here).
    show("case 1, level:", computeHandPoseError(fk, Eigen::Quaternionf(1, 0, 0, 0), target, targetQuat, 0.0f));

    // Case 2: pelvis pitched 5 degrees forward. Same joints, so the palms really did
    // tilt with the pelvis and the errors change.
    Eigen::Quaternionf pitch(std::cos(0.0436332f), 0, std::sin(0.0436332f), 0);  // 5 deg about y
    show("case 2, pitched 5 deg forward:", computeHandPoseError(fk, pitch, target, targetQuat, 0.0f));

    // Case 3: only the yaw changed (90 degrees): nothing should change vs case 1,
    // because the yaw is removed.
    Eigen::Quaternionf yaw(std::cos(0.7853982f), 0, 0, std::sin(0.7853982f));  // 90 deg about z
    show("case 3, yaw 90 deg (must equal case 1):", computeHandPoseError(fk, yaw, target, targetQuat, 0.0f));
    return 0;
}
