
#include <iostream>
#include <map>
#include <string>

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
    unsigned long p = 5;
    unsigned long m = 8192;
    unsigned long r = 1;
    unsigned long bits = 72;
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

	// Encrypt the inputs
	std::vector<long> vec_x(1, extract_input("x"));
	ptxt_t ptxt_x(context, vec_x);
	ctxt_t ciph_x(public_key);
	public_key.Encrypt(ciph_x, ptxt_x);
	std::vector<long> vec_y(1, extract_input("y"));
	ptxt_t ptxt_y(context, vec_y);
	ctxt_t ciph_y(public_key);
	public_key.Encrypt(ciph_y, ptxt_y);

	// Perform the actual circuit
	ctxt_t stack_0 = ciph_x;
	ctxt_t stack_1 = ciph_y;
	stack_1 *= 4l;
	stack_0 += stack_1;
	stack_0 *= stack_0;
	stack_0 *= stack_0;
	stack_0 *= 4l;
	stack_0 += 1l;
	ptxt_t decrypted(context);
	secret_key.Decrypt(decrypted, stack_0);
	std::cout << decrypted << std::endl;

    return 0;
}
