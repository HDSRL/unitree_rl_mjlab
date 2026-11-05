// multi_mocap_bridge.cpp
// Build: g++ -std=c++17 multi_mocap_bridge.cpp -o multi_mocap_bridge -lvrpn -lpthread
// Usage: ./multi_mocap_bridge <server_ip:port> <udp_ip> <udp_port> <hz> <obj1> <obj2> <obj3> ...

#include <vrpn_Tracker.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <memory>

static std::atomic<bool> g_running(true);

double now_sec() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

struct Vec3 { double x{0}, y{0}, z{0}; };
struct Quat { double w{1}, x{0}, y{0}, z{0}; };

// --- Math Helpers ---
static Quat quat_normalize(const Quat& q) {
    double n = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    return (n < 1e-12) ? Quat{} : Quat{q.w/n, q.x/n, q.y/n, q.z/n};
}

static Quat quat_mul(const Quat& a, const Quat& b) {
    return { a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
             a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
             a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
             a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w };
}

static Vec3 angular_velocity_from_quat(const Quat& q_prev, const Quat& q_curr, double dt) {
    if (dt <= 1e-6) return {0,0,0};
    Quat dq = quat_normalize(quat_mul({q_prev.w, -q_prev.x, -q_prev.y, -q_prev.z}, q_curr));
    if (dq.w < 0.0) { dq.w = -dq.w; dq.x = -dq.x; dq.y = -dq.y; dq.z = -dq.z; }
    double sin_half = std::sqrt(dq.x*dq.x + dq.y*dq.y + dq.z*dq.z);
    double angle = 2.0 * std::atan2(sin_half, dq.w);
    if (sin_half < 1e-9) return {0,0,0};
    double s = angle / (sin_half * dt);
    return {dq.x * s, dq.y * s, dq.z * s};
}

// --- Data Structures ---
#pragma pack(push, 1)
struct UdpPacket {
    uint32_t magic{0x4D4F4341};
    uint32_t id;    // Unique ID for the object
    uint32_t seq;
    double t;
    double px, py, pz;
    double qw, qx, qy, qz;
    double vx, vy, vz;
    double wx, wy, wz;
};
#pragma pack(pop)

struct TrackedObject {
    uint32_t id;
    std::string name;
    std::unique_ptr<vrpn_Tracker_Remote> tracker;
    
    std::mutex mtx;
    UdpPacket last_pkt{};
    bool has_data{false};

    // Velocity calc state
    bool has_prev{false};
    double prev_t;
    Vec3 prev_p;
    Quat prev_q;
    uint32_t seq_count{0};

    TrackedObject(uint32_t _id, std::string _name, std::string _addr) 
        : id(_id), name(_name) {
        tracker = std::make_unique<vrpn_Tracker_Remote>((name + "@" + _addr).c_str());
    }
};

// --- Bridge Class ---
class MultiMocapBridge {
public:
    MultiMocapBridge(std::string server, std::string ip, int port, double rate)
        : server_addr_(server), udp_ip_(ip), udp_port_(port), rate_hz_(rate) {}

    bool add_object(std::string name) {
        uint32_t id = objects_.size();
        auto obj = std::make_unique<TrackedObject>(id, name, server_addr_);
        obj->tracker->register_change_handler(obj.get(), handle_vrpn);
        objects_.push_back(std::move(obj));
        return true;
    }

    void run() {
        int fd = socket(AF_INET, SOCK_DGRAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(udp_port_);
        inet_pton(AF_INET, udp_ip_.c_str(), &addr.sin_addr);

        std::thread udp_thread([&]() {
            auto next_tick = std::chrono::steady_clock::now();
            while (g_running) {
                for (auto& obj : objects_) {
                    std::lock_guard<std::mutex> lock(obj->mtx);
                    if (obj->has_data) {
                        sendto(fd, &obj->last_pkt, sizeof(UdpPacket), 0, (sockaddr*)&addr, sizeof(addr));
                    }
                }
                next_tick += std::chrono::microseconds((int)(1e6 / rate_hz_));
                std::this_thread::sleep_until(next_tick);
            }
        });

        std::cout << "Bridge running for " << objects_.size() << " objects...\n";
        while (g_running) {
            for (auto& obj : objects_) obj->tracker->mainloop();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (udp_thread.joinable()) udp_thread.join();
        close(fd);
    }

private:
    static void VRPN_CALLBACK handle_vrpn(void* userdata, const vrpn_TRACKERCB t) {
        auto* obj = static_cast<TrackedObject*>(userdata);
        double now = now_sec();
        
        std::lock_guard<std::mutex> lock(obj->mtx);
        obj->last_pkt.id = obj->id;
        obj->last_pkt.t = now;
        obj->last_pkt.seq = ++obj->seq_count;
        obj->last_pkt.px = t.pos[0]; obj->last_pkt.py = t.pos[1]; obj->last_pkt.pz = t.pos[2];
        obj->last_pkt.qw = t.quat[3]; obj->last_pkt.qx = t.quat[0]; 
        obj->last_pkt.qy = t.quat[1]; obj->last_pkt.qz = t.quat[2];

        if (obj->has_prev) {
            double dt = now - obj->prev_t;
            if (dt > 0.001) {
                obj->last_pkt.vx = (obj->last_pkt.px - obj->prev_p.x) / dt;
                obj->last_pkt.vy = (obj->last_pkt.py - obj->prev_p.y) / dt;
                obj->last_pkt.vz = (obj->last_pkt.pz - obj->prev_p.z) / dt;
                Vec3 w = angular_velocity_from_quat(obj->prev_q, {obj->last_pkt.qw, obj->last_pkt.qx, obj->last_pkt.qy, obj->last_pkt.qz}, dt);
                obj->last_pkt.wx = w.x; obj->last_pkt.wy = w.y; obj->last_pkt.wz = w.z;
                obj->has_data = true;
            }
        }
        obj->prev_t = now;
        obj->prev_p = {obj->last_pkt.px, obj->last_pkt.py, obj->last_pkt.pz};
        obj->prev_q = {obj->last_pkt.qw, obj->last_pkt.qx, obj->last_pkt.qy, obj->last_pkt.qz};
        obj->has_prev = true;
    }

    std::string server_addr_, udp_ip_;
    int udp_port_;
    double rate_hz_;
    std::vector<std::unique_ptr<TrackedObject>> objects_;
};

void sig_handler(int) { g_running = false; }

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <server_ip:port> <udp_ip> <udp_port> <hz> <obj1> [obj2...]\n";
        return 1;
    }
    std::signal(SIGINT, sig_handler);
    MultiMocapBridge bridge(argv[1], argv[2], std::stoi(argv[3]), std::stod(argv[4]));
    for (int i = 5; i < argc; ++i) bridge.add_object(argv[i]);
    bridge.run();
    return 0;
}
