#pragma once

#include <cstdint>  
#include <mutex>
#include <array>
#include <cstddef>

// Struct must match the bridge exactly
#pragma pack(push, 1)
struct UdpPacket {
    uint32_t magic;
    uint32_t id;    
    uint32_t seq;
    double t;
    double px, py, pz;
    double qw, qx, qy, qz;
    double vx, vy, vz;
    double wx, wy, wz;
};
#pragma pack(pop)

constexpr std::size_t NumMocapBodiesPerAgent = 3;//torso, left foot, right foot

struct MocapData{
    std::mutex mutex_;
    std::array<UdpPacket, NumMocapBodiesPerAgent> packets{}; //{} zeros all fields

};