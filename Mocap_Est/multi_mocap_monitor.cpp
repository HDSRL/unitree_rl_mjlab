// mocap_monitor.cpp
// Build: g++ -std=c++17 mocap_monitor.cpp -o mocap_monitor
// Usage: ./mocap_monitor 15000

#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <thread>

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

int main(int argc, char** argv) {
    int port = (argc > 1) ? std::stoi(argv[1]) : 15000;

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) { perror("socket"); return 1; }

    // Set a timeout so we can refresh the UI even if no data comes in
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 1000; // 1ms
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }

    // Map to store the latest data for each unique ID
    std::map<uint32_t, UdpPacket> registry;
    
    std::cout << "Monitoring port " << port << " for Mocap IDs..." << std::endl;

    UdpPacket pkt;
    auto last_print = std::chrono::steady_clock::now();

while (true) {
    // Drain all currently queued packets.
    while (true) {
        ssize_t n = recv(fd, &pkt, sizeof(pkt), MSG_DONTWAIT);
        if (n == sizeof(UdpPacket) && pkt.magic == 0x4D4F4341) {
            registry[pkt.id] = pkt;
        } else {
            break;
        }
    }

    auto now = std::chrono::steady_clock::now();
    if (now - last_print >= std::chrono::milliseconds(50)) {
        last_print = now;

        std::cout << "\033[2J\033[H";
        std::cout << "================================================================\n";
        std::cout << " LIVE MOCAP MONITOR | Port: " << port << "\n";
        std::cout << "================================================================\n";
        std::cout << std::left << std::setw(6)  << "ID"
                  << std::setw(10) << "Sequence"
                  << std::setw(25) << "Position (X, Y, Z)"
                  << "Time (s)\n";
        std::cout << "----------------------------------------------------------------\n";

        for (auto const& [id, data] : registry) {
            std::cout << std::left << std::setw(6) << id
                      << std::setw(10) << data.seq
                      << "(" << std::fixed << std::setprecision(3)
                      << std::setw(6) << data.px << ", "
                      << std::setw(6) << data.py << ", "
                      << std::setw(6) << data.pz << ")   "
                      << std::setprecision(2) << data.t << "\n";
        }

        if (registry.empty()) {
            std::cout << "  (Waiting for data... Ensure bridge is sending to this IP)\n";
        }

        std::cout << std::flush;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}


    close(fd);
    return 0;
}
