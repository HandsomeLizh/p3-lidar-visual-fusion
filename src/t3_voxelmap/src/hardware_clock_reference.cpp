// Pair a device timestamp with the local driver's DDS publication timestamp.
// Humble's Python executor does not expose MessageInfo. Keep this small reader
// in C++ so Python scheduling/image queues cannot alter the clock estimate.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

class ClockReference : public rclcpp::Node {
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr input_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr output_;
public:
    ClockReference():Node("p3_hardware_clock_reference") {
        auto topic=declare_parameter<std::string>("imu_topic","/Car/T5/OS1/imu");
        auto output=declare_parameter<std::string>("output_topic","/P3/hardware/clock_reference");
        output_=create_publisher<std_msgs::msg::Float64MultiArray>(output,rclcpp::QoS(200).reliable());
        input_=create_subscription<sensor_msgs::msg::Imu>(topic,rclcpp::SensorDataQoS().keep_last(200),
            [this](sensor_msgs::msg::Imu::ConstSharedPtr m,const rclcpp::MessageInfo &info) {
                const auto ns=info.get_rmw_message_info().source_timestamp;
                if(ns<=0) {
                    RCLCPP_ERROR_THROTTLE(get_logger(),*get_clock(),5000,"DDS source timestamp unavailable; clock reference rejected");
                    return;
                }
                std_msgs::msg::Float64MultiArray pair;
                pair.data={rclcpp::Time(m->header.stamp).seconds(),double(ns)*1e-9};
                output_->publish(pair);
            });
    }
};
int main(int argc,char **argv) {
    rclcpp::init(argc,argv);auto node=std::make_shared<ClockReference>();
    rclcpp::spin(node);rclcpp::shutdown();return 0;
}
