
#include <iostream>
#include <map>
#include <string>
#include <chrono>

#include <helib/helib.h>

typedef helib::Ptxt<helib::BGV> ptxt_t;
typedef helib::Ctxt ctxt_t;

std::map<std::string, int> input_map;

void parse_arguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string argument(argv[i]);
        size_t pos = argument.find('=');
        if (pos != std::string::npos) {
            std::string key = argument.substr(0, pos);
            int value = std::stoi(argument.substr(pos + 1));
            input_map[key] = value;
        }
    }
}

int extract_input(const std::string& name) {
    if (input_map.find(name) != input_map.end()) {
        return input_map[name];
    } else {
        std::cerr << "Error: " << name << " not found" << std::endl;
        return -1;
    }
}

int main(int argc, char* argv[]) {
    // Parse the inputs
    parse_arguments(argc, argv);

    // Set up the HE parameters
    unsigned long p = 257;
    unsigned long m = 65536;
    unsigned long r = 1;
    unsigned long bits = 449;
    unsigned long c = 3;
    helib::Context context = helib::ContextBuilder<helib::BGV>()
        .m(m)
        .p(p)
        .r(r)
        .bits(bits)
        .c(c)
        .build();


    // Generate keys
    helib::SecKey secret_key(context);
    secret_key.GenSecKey();
    helib::addSome1DMatrices(secret_key);
    const helib::PubKey& public_key = secret_key;

