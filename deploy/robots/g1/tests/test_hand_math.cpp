// Step-2 check of the two math helpers in hand_observations.hpp. Run it from build/.
#include "hand_observations.hpp"
#include <cmath>
#include <iostream>

int main()
{
    // 6D of a 90 degree turn about z: x axis -> y axis, y axis -> -x axis.
    float a = M_PI / 2;
    Eigen::Quaternionf turn(std::cos(a / 2), 0, 0, std::sin(a / 2));  // (w, x, y, z)
    std::vector<float> six;
    isaaclab::mdp::append6D(turn, six);
    std::cout << "6D:";
    for (float v : six) std::cout << " " << v;
    std::cout << "   expected: 0 1 0 -1 0 0 (rounded)\n";

    // Glide a target from identity toward a 1.0 rad turn about z, 0.1 rad per step.
    Eigen::Quaternionf q(1, 0, 0, 0);  // identity
    Eigen::Quaternionf goal(std::cos(0.5f), 0, 0, std::sin(0.5f));  // 1.0 rad about z
    std::cout << "angle to the goal after each step (expected 0.9, 0.8, ..., 0.0):\n";
    for (int i = 0; i < 12; i++) {
        q = isaaclab::mdp::glideQuat(q, goal, 0.1f);
        std::cout << " " << q.angularDistance(goal);
    }
    std::cout << "\n";
    return 0;
}
