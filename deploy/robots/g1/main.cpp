#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>   
#include <unistd.h>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();
std::shared_ptr<MocapData> FSMState::mocap_data = std::make_shared<MocapData>();

int mocap_port = 15000;
int udpSock = -1;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    //Redcue the timeout to 200 ms for faster detection of robot connection loss.
    FSMState::lowstate->set_timeout_ms(200);
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}


void InitMocapUdp()
{
    udpSock = socket(AF_INET, SOCK_DGRAM, 0);
    if (udpSock < 0) {
        std::cerr << "[parent mocap] Failed to create UDP socket." << std::endl;
        return;
    }

    int opt = 1;
    setsockopt(udpSock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 10000;
    setsockopt(udpSock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(mocap_port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(udpSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        spdlog::error("Mocap: failed to bind UDP port {}. No mocap data this run.", mocap_port);
        close(udpSock);
        udpSock = -1;
        return;
    }

    spdlog::info("[parent mocap] Listening on UDP port {}", mocap_port);
}

void ReadUdpPacket()
{
    
    if (udpSock < 0) {
        // spdlog::error("[Mocap] UDP socket is not initialized.");
        return;
        
    }

    UdpPacket pkt{};
    const ssize_t n = recv(udpSock, &pkt, sizeof(pkt), 0);

    if (n < 0) {
        return;
    }

    if (n != static_cast<ssize_t>(sizeof(UdpPacket)) || pkt.magic != 0x4D4F4341) {
        return;
    }

    if (pkt.id < NumMocapBodiesPerAgent) {
        std::lock_guard<std::mutex> lock(FSMState::mocap_data->mutex_);        
        FSMState::mocap_data->packets[pkt.id] = pkt;
    }    


    // if (udpSock >= 0) {
    //     close(udpSock);
    //     udpSock = -1;
    // }
}



int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 6; // 29dof
    // 6 = our G1: 29 motors, waist roll/pitch locked
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    //Mocap UDP initialization
    InitMocapUdp();
    auto mocap_thread = unitree::common::CreateRecurrentThreadEx("MocapUDP_loop", 19, 1000, &ReadUdpPacket);

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + A] to start controlling the robot.\n";
    // std::cout << "And then press [R1 + A/B/Y/X] to control the robot dance.\n";
    std::array<uint32_t, NumMocapBodiesPerAgent> last_seq{};
    while (true)
    {
        sleep(1);
        std::array<UdpPacket, NumMocapBodiesPerAgent> packets{};
        {
            std::lock_guard<std::mutex> lock(FSMState::mocap_data->mutex_);
            packets = FSMState::mocap_data->packets;
        }

        for (std::size_t i = 0; i < packets.size(); ++i) {
            if (packets[i].seq == last_seq[i]) {
                spdlog::warn("Mocap: body {} has no new data", i);
            }
            last_seq[i] = packets[i].seq;
        }
    }
    
    return 0;
}

