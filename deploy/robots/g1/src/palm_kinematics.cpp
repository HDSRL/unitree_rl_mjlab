#include "palm_kinematics.hpp"
#include <cstdlib>
#include <iostream>

PalmKinematics::PalmKinematics(std::string xmlPath, std::vector<std::string> jointNames)
{
    // Load the robot description: the same XML training used.
    char error[1000] = "";
    model = mj_loadXML(xmlPath.c_str(), nullptr, error, 1000);
    if (model == nullptr) {
        std::cerr << "PalmKinematics: cannot load " << xmlPath << ": " << error << std::endl;
        exit(1);
    }
    data = mj_makeData(model);

    // data->qpos is one flat array: [base x y z, base quaternion w x y z, joint angles].
    // We ask MuJoCo where each joint's angle lives instead of assuming an order.
    for (int i = 0; i < (int)jointNames.size(); i++) {
        int jointId = mj_name2id(model, mjOBJ_JOINT, jointNames[i].c_str());
        if (jointId < 0) {
            std::cerr << "PalmKinematics: no joint named " << jointNames[i] << std::endl;
            exit(1);
        }
        qposIndex.push_back(model->jnt_qposadr[jointId]);
    }

    palmSite[0] = findSite("left_palm");
    palmSite[1] = findSite("right_palm");
    footSite[0] = findSite("left_foot");
    footSite[1] = findSite("right_foot");
}

PalmKinematics::~PalmKinematics()
{
    // MuJoCo allocated these, so MuJoCo frees them.
    mj_deleteData(data);
    mj_deleteModel(model);
}

void PalmKinematics::compute(const Eigen::VectorXf& jointPos)
{
    // 1. Pelvis at the origin and not rotated, so MuJoCo's "world" IS the pelvis frame.
    mju_zero(data->qpos, model->nq);
    data->qpos[3] = 1.0;  // quaternion w = 1 means "no rotation"

    // 2. Each joint angle into its own slot.
    for (int i = 0; i < (int)qposIndex.size(); i++) {
        data->qpos[qposIndex[i]] = jointPos[i];
    }

    // 3. Forward kinematics: positions and orientations of every body and site.
    //    Kinematics only (no forces, no contacts), so it takes microseconds.
    mj_kinematics(model, data);

    // 4. Read out the palm and foot sites.
    for (int s = 0; s < 2; s++) {
        palmPos[s] = sitePosition(palmSite[s]);
        palmRot[s] = siteRotation(palmSite[s]);
        footPos[s] = sitePosition(footSite[s]);
    }
}

int PalmKinematics::findSite(std::string name)
{
    int site = mj_name2id(model, mjOBJ_SITE, name.c_str());
    if (site < 0) {
        std::cerr << "PalmKinematics: no site named " << name << std::endl;
        exit(1);
    }
    return site;
}

Eigen::Vector3d PalmKinematics::sitePosition(int site)
{
    // site_xpos stores 3 numbers per site, one site after another.
    double* p = data->site_xpos + 3 * site;
    return Eigen::Vector3d(p[0], p[1], p[2]);
}

Eigen::Matrix3d PalmKinematics::siteRotation(int site)
{
    // site_xmat stores 9 numbers per site, ROW by row:
    // element (row, col) is at index 3 * row + col.
    double* m = data->site_xmat + 9 * site;
    Eigen::Matrix3d R;
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            R(row, col) = m[3 * row + col];
        }
    }
    return R;
}
