g++ Broadcast_node.cpp -o broadcast_node -I/usr/local/include -L/usr/local/lib -lvrpn -lquat
broadcast_node "gauge_case@128.173.27.71:3883" 127.0.0.1 15000 100


g++ MultiBroadcast_node.cpp -o multi_mocap_bridge -I/usr/local/include -L/usr/local/lib -lvrpn -lquat
./multi_mocap_bridge 128.173.27.71:3883 127.0.0.1 15000 100 <Obj1_name> <Obj2_name> 

g++ multi_mocap_monitor.cpp -o monitor_node
./monitor_node [Optional <port>]



Names
A1_willow
Go2_Tom
Go2_Jerry

