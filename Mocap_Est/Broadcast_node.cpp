// mocap_vrpn_udp_bridge.cpp
//
// Build example:
//   g++ -std=c++17 mocap_vrpn_udp_bridge.cpp -o mocap_bridge \
//       -lvrpn -lpthread
//
// If VRPN headers/libs are not in default path, add include/library paths, e.g.
//   g++ -std=c++17 mocap_vrpn_udp_bridge.cpp -o mocap_bridge \
//       -I/usr/include -L/usr/lib -lvrpn -lpthread
//
// Usage example:
//   ./mocap_bridge "Base@192.168.1.100:3883" 127.0.0.1 15000 100
//
// Arguments:
//   argv[1] = VRPN tracker address, e.g. "RigidBodyName@server_ip:3883"
//   argv[2] = UDP destination IP
//   argv[3] = UDP destination port
//   argv[4] = UDP publish rate in Hz
//
// Notes:
// - Position is world frame.
// - Quaternion is [w x y z].
// - Linear velocity is world frame.
// - Angular velocity here is approximated from quaternion difference.
// - UDP packet is sent in binary using the struct below.
// - Keep Motive rigid body name exactly matched.

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

static std::atomic<bool> g_running(true);

double now_sec()
{
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    auto dt = clock::now() - t0;
    return std::chrono::duration<double>(dt).count();
}

struct Vec3
{
    double x{0}, y{0}, z{0};
};

struct Quat
{
    // stored as w, x, y, z
    double w{1}, x{0}, y{0}, z{0};
};

static Quat quat_normalize(const Quat& q)
{
    double n = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (n < 1e-12) return Quat{};
    return Quat{q.w/n, q.x/n, q.y/n, q.z/n};
}

static Quat quat_conj(const Quat& q)
{
    return Quat{q.w, -q.x, -q.y, -q.z};
}

static Quat quat_mul(const Quat& a, const Quat& b)
{
    return Quat{
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

static Vec3 vec_sub(const Vec3& a, const Vec3& b)
{
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vec3 vec_scale(const Vec3& a, double s)
{
    return Vec3{a.x*s, a.y*s, a.z*s};
}

// Convert relative quaternion to rotation vector / dt.
// Small and practical version for mocap usage.
static Vec3 angular_velocity_from_quat(const Quat& q_prev, const Quat& q_curr, double dt)
{
    if (dt <= 1e-6) return Vec3{0,0,0};

    Quat dq = quat_mul(quat_conj(q_prev), q_curr);
    dq = quat_normalize(dq);

    // Keep shortest rotation
    if (dq.w < 0.0) {
        dq.w = -dq.w;
        dq.x = -dq.x;
        dq.y = -dq.y;
        dq.z = -dq.z;
    }

    double sin_half = std::sqrt(dq.x*dq.x + dq.y*dq.y + dq.z*dq.z);
    double angle = 2.0 * std::atan2(sin_half, dq.w);

    if (sin_half < 1e-9 || std::abs(angle) < 1e-9) {
        return Vec3{0,0,0};
    }

    Vec3 axis{dq.x / sin_half, dq.y / sin_half, dq.z / sin_half};
    Vec3 rotvec = vec_scale(axis, angle);

    return vec_scale(rotvec, 1.0 / dt);
}

struct MocapState
{
    double t{0.0};
    uint32_t seq{0};

    Vec3 p;    // world position
    Quat q;    // world orientation, wxyz
    Vec3 v;    // world linear velocity
    Vec3 w;    // angular velocity
    bool valid{false};
};

#pragma pack(push, 1)
struct UdpPacket
{
    uint32_t magic;   // 0x4D4F4341 = "MOCA"
    uint32_t seq;
    double t;

    double px, py, pz;
    double qw, qx, qy, qz;
    double vx, vy, vz;
    double wx, wy, wz;
};
#pragma pack(pop)

class MocapBridge
{
public:
    MocapBridge(const std::string& vrpn_addr,
                const std::string& udp_ip,
                int udp_port,
                double udp_rate_hz)
        : tracker_addr_(vrpn_addr), udp_ip_(udp_ip), udp_port_(udp_port), udp_rate_hz_(udp_rate_hz)
    {
    }

    bool init()
    {
        tracker_ = new vrpn_Tracker_Remote(tracker_addr_.c_str());
        if (!tracker_) {
            std::cerr << "Failed to create VRPN tracker\n";
            return false;
        }

        tracker_->register_change_handler(this, handle_tracker);

        sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) {
            std::perror("socket");
            return false;
        }

        std::memset(&udp_addr_, 0, sizeof(udp_addr_));
        udp_addr_.sin_family = AF_INET;
        udp_addr_.sin_port = htons(static_cast<uint16_t>(udp_port_));

        if (inet_pton(AF_INET, udp_ip_.c_str(), &udp_addr_.sin_addr) != 1) {
            std::cerr << "Invalid UDP IP: " << udp_ip_ << "\n";
            return false;
        }

        std::cout << "VRPN tracker: " << tracker_addr_ << "\n";
        std::cout << "UDP target  : " << udp_ip_ << ":" << udp_port_ << "\n";
        std::cout << "UDP rate    : " << udp_rate_hz_ << " Hz\n";

        return true;
    }

    void run()
    {
        std::thread udp_thread(&MocapBridge::udp_loop, this);
        auto last_callback_warn_tp = std::chrono::steady_clock::now();

        while (g_running.load()) {
            tracker_->mainloop();

            auto now_tp = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(mtx_);
                if (callback_count_ == 0 &&
                    std::chrono::duration_cast<std::chrono::seconds>(now_tp - start_tp_).count() >= 5 &&
                    std::chrono::duration_cast<std::chrono::seconds>(now_tp - last_callback_warn_tp).count() >= 5) {
                    std::cerr << "[VRPN] waiting for tracker callbacks from " << tracker_addr_ << "\n";
                    last_callback_warn_tp = now_tp;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (udp_thread.joinable()) udp_thread.join();
    }

    ~MocapBridge()
    {
        if (tracker_) {
            delete tracker_;
            tracker_ = nullptr;
        }
        if (sockfd_ >= 0) {
            close(sockfd_);
            sockfd_ = -1;
        }
    }

private:
    static void VRPN_CALLBACK handle_tracker(void* userdata, const vrpn_TRACKERCB t)
    {
        auto* self = static_cast<MocapBridge*>(userdata);
        self->on_tracker(t);
    }

    void on_tracker(const vrpn_TRACKERCB& t)
    {
        double ts = now_sec();

        MocapState s;
        s.t = ts;
        s.seq = ++seq_counter_;

        s.p = Vec3{t.pos[0], t.pos[1], t.pos[2]};

        // VRPN quaternion is commonly [x y z w]
        s.q = quat_normalize(Quat{t.quat[3], t.quat[0], t.quat[1], t.quat[2]});

        if (has_prev_) {
            double dt = ts - prev_t_;
            if (dt > 1e-4 && dt < 0.2) {
                s.v = vec_scale(vec_sub(s.p, prev_p_), 1.0 / dt);
                s.w = angular_velocity_from_quat(prev_q_, s.q, dt);
                s.valid = true;
            }
        }

        prev_t_ = ts;
        prev_p_ = s.p;
        prev_q_ = s.q;
        has_prev_ = true;

        {
            std::lock_guard<std::mutex> lock(mtx_);
            latest_ = s;
            ++callback_count_;
            if (callback_count_ <= 5 || (callback_count_ % 50) == 0) {
                std::cout << std::fixed << std::setprecision(4)
                          << "[VRPN] callback #" << callback_count_
                          << " p=(" << s.p.x << ", " << s.p.y << ", " << s.p.z << ")"
                          << " q=(" << s.q.w << ", " << s.q.x << ", " << s.q.y << ", " << s.q.z << ")"
                          << "\n";
            }
        }
    }

    void udp_loop()
    {
        using namespace std::chrono;
        const double period = 1.0 / udp_rate_hz_;
        auto next_tp = steady_clock::now();

        while (g_running.load()) {
            MocapState s;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                s = latest_;
            }

            if (s.valid) {
                UdpPacket pkt{};
                pkt.magic = 0x4D4F4341; // "MOCA"
                pkt.seq   = s.seq;
                pkt.t     = s.t;

                pkt.px = s.p.x;
                pkt.py = s.p.y;
                pkt.pz = s.p.z;

                pkt.qw = s.q.w;
                pkt.qx = s.q.x;
                pkt.qy = s.q.y;
                pkt.qz = s.q.z;

                pkt.vx = s.v.x;
                pkt.vy = s.v.y;
                pkt.vz = s.v.z;

                pkt.wx = s.w.x;
                pkt.wy = s.w.y;
                pkt.wz = s.w.z;

                ssize_t n = sendto(sockfd_,
                                   &pkt,
                                   sizeof(pkt),
                                   0,
                                   reinterpret_cast<sockaddr*>(&udp_addr_),
                                   sizeof(udp_addr_));

                if (n < 0) {
                    std::perror("sendto");
                }

                // lightweight console debug
                static uint32_t last_print_seq = 0;
                if (pkt.seq - last_print_seq >= 50) {
                    last_print_seq = pkt.seq;
                    std::cout << std::fixed << std::setprecision(4)
                              << "[UDP] seq=" << pkt.seq
                              << " p=(" << pkt.px << ", " << pkt.py << ", " << pkt.pz << ")"
                              << " v=(" << pkt.vx << ", " << pkt.vy << ", " << pkt.vz << ")"
                              << "\n";
                }
            }

            next_tp += duration_cast<steady_clock::duration>(duration<double>(period));
            std::this_thread::sleep_until(next_tp);
        }
    }

private:
    std::string tracker_addr_;
    std::string udp_ip_;
    int udp_port_{15000};
    double udp_rate_hz_{100.0};

    vrpn_Tracker_Remote* tracker_{nullptr};

    int sockfd_{-1};
    sockaddr_in udp_addr_{};

    std::mutex mtx_;
    MocapState latest_{};
    std::chrono::steady_clock::time_point start_tp_{std::chrono::steady_clock::now()};

    bool has_prev_{false};
    double prev_t_{0.0};
    Vec3 prev_p_{};
    Quat prev_q_{};

    std::atomic<uint32_t> seq_counter_{0};
    uint32_t callback_count_{0};
};

void signal_handler(int)
{
    g_running.store(false);
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <vrpn_tracker@server:port> <udp_ip> <udp_port> <udp_rate_hz>\n\n"
            << "Example:\n  "
            << argv[0]
            << " \"Base@192.168.1.100:3883\" 127.0.0.1 15000 100\n";
        return 1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    const std::string vrpn_addr = argv[1];
    const std::string udp_ip    = argv[2];
    const int udp_port          = std::stoi(argv[3]);
    const double udp_rate_hz    = std::stod(argv[4]);

    MocapBridge bridge(vrpn_addr, udp_ip, udp_port, udp_rate_hz);
    if (!bridge.init()) {
        return 1;
    }

    bridge.run();
    return 0;
}