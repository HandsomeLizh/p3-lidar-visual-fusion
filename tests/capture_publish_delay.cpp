// Test-only LD_PRELOAD shim. Never installed or used against live sensor domains.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <thread>
#include <rmw/rmw.h>
extern "C" rmw_ret_t rmw_publish(const rmw_publisher_t* publisher, const void* message, rmw_publisher_allocation_t* allocation) {
    using Publish = decltype(&rmw_publish);
    static const auto original = reinterpret_cast<Publish>(dlsym(RTLD_NEXT, "rmw_publish"));
    static std::atomic<int> writes{0};
    const char* delay = std::getenv("P3_TEST_WRITE_DELAY_SEC");
    if (delay && std::strcmp(publisher->topic_name, "/test/left/image_raw") == 0 && ++writes == 2)
        std::this_thread::sleep_for(std::chrono::duration<double>(std::atof(delay)));
    return original(publisher, message, allocation);
}
