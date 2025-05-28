
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <chrono>

#include "openfhe.h"

using namespace lbcrypto;

typedef lbcrypto::Plaintext ptxt_t;
typedef lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ctxt_t;

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
    unsigned long p = 786433;
    unsigned long m = 1 << 17;
    //unsigned long r = 1;
    //unsigned long bits = 134;
    //unsigned long c = 3;
    CCParams<CryptoContextBFVRNS> parameters;
    parameters.SetPlaintextModulus(p);
    parameters.SetRingDim(m);
    parameters.SetMultiplicativeDepth(5);
    parameters.SetBatchSize(m / 2);
    CryptoContext<DCRTPoly> context = GenCryptoContext(parameters);


    context->Enable(lbcrypto::PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);

    // Generate keys
    auto keys = context->KeyGen();
    context->EvalMultKeyGen(keys.secretKey);
    context->EvalSumKeyGen(keys.secretKey);
    auto& public_key = keys.publicKey;

	// Encrypt the inputs
	std::vector<int64_t> vec_x(1, extract_input("x"));
	ptxt_t ptxt_x = context->MakePackedPlaintext(vec_x);
	ctxt_t ciph_x = context->Encrypt(public_key, ptxt_x);
	std::vector<int64_t> vec_y(1, extract_input("y"));
	ptxt_t ptxt_y = context->MakePackedPlaintext(vec_y);
	ctxt_t ciph_y = context->Encrypt(public_key, ptxt_y);

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < 10; i++) {
	// Perform the actual circuit
	ctxt_t stack_0 = ciph_x;
	ctxt_t stack_1 = context->EvalAdd(stack_0, context->MakePackedPlaintext({ int64_t(4) }));
	ctxt_t stack_2 = context->EvalMultAndRelinearize(stack_1, stack_1);
	ctxt_t stack_3 = context->EvalMult(stack_2, context->MakePackedPlaintext({ int64_t(6) }));
	context->EvalAddInPlace(stack_3, context->MakePackedPlaintext({ int64_t(4) }));
	ctxt_t stack_4 = context->EvalMultAndRelinearize(stack_2, stack_2);
	context->EvalAddInPlace(stack_3, stack_4);
	stack_3 = context->EvalMult(stack_3, context->MakePackedPlaintext({ int64_t(6) }));
	stack_1 = context->EvalMultAndRelinearize(stack_1, stack_3);
	stack_2 = context->EvalMultAndRelinearize(stack_2, stack_4);
	stack_2 = context->EvalMult(stack_2, context->MakePackedPlaintext({ int64_t(4) }));
	context->EvalAddInPlace(stack_1, stack_2);
	stack_2 = ciph_y;
	stack_3 = context->EvalAdd(stack_2, context->MakePackedPlaintext({ int64_t(4) }));
	stack_4 = context->EvalMultAndRelinearize(stack_3, stack_3);
	ctxt_t stack_5 = context->EvalMult(stack_4, context->MakePackedPlaintext({ int64_t(6) }));
	context->EvalAddInPlace(stack_5, context->MakePackedPlaintext({ int64_t(4) }));
	ctxt_t stack_6 = context->EvalMultAndRelinearize(stack_4, stack_4);
	context->EvalAddInPlace(stack_5, stack_6);
	stack_5 = context->EvalMult(stack_5, context->MakePackedPlaintext({ int64_t(6) }));
	stack_3 = context->EvalMultAndRelinearize(stack_3, stack_5);
	stack_4 = context->EvalMultAndRelinearize(stack_4, stack_6);
	stack_4 = context->EvalMult(stack_4, context->MakePackedPlaintext({ int64_t(4) }));
	context->EvalAddInPlace(stack_3, stack_4);
	stack_4 = context->EvalMultAndRelinearize(stack_1, stack_3);
	stack_5 = context->EvalMult(stack_1, context->MakePackedPlaintext({ int64_t(6) }));
	context->EvalAddInPlace(stack_5, context->MakePackedPlaintext({ int64_t(1) }));
	stack_6 = context->EvalMult(stack_3, context->MakePackedPlaintext({ int64_t(6) }));
	context->EvalAddInPlace(stack_6, context->MakePackedPlaintext({ int64_t(1) }));
	stack_5 = context->EvalMultAndRelinearize(stack_5, stack_6);
	context->EvalAddInPlace(stack_4, stack_5);
	stack_2 = context->EvalMult(stack_2, context->MakePackedPlaintext({ int64_t(6) }));
	context->EvalAddInPlace(stack_0, stack_2);
	stack_2 = context->EvalMultAndRelinearize(stack_0, stack_0);
	stack_5 = context->EvalMult(stack_2, context->MakePackedPlaintext({ int64_t(6) }));
	context->EvalAddInPlace(stack_5, context->MakePackedPlaintext({ int64_t(4) }));
	stack_6 = context->EvalMultAndRelinearize(stack_2, stack_2);
	context->EvalAddInPlace(stack_5, stack_6);
	stack_5 = context->EvalMult(stack_5, context->MakePackedPlaintext({ int64_t(6) }));
	stack_0 = context->EvalMultAndRelinearize(stack_0, stack_5);
	stack_2 = context->EvalMultAndRelinearize(stack_2, stack_6);
	stack_2 = context->EvalMult(stack_2, context->MakePackedPlaintext({ int64_t(4) }));
	context->EvalAddInPlace(stack_0, stack_2);
	stack_0 = context->EvalMultAndRelinearize(stack_4, stack_0);
	stack_2 = context->EvalMult(stack_3, context->MakePackedPlaintext({ int64_t(6) }));
	context->EvalAddInPlace(stack_2, context->MakePackedPlaintext({ int64_t(1) }));
	stack_1 = context->EvalMultAndRelinearize(stack_2, stack_1);
	context->EvalAddInPlace(stack_0, stack_1);
	std::cout << "Done!" << std::endl;
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << elapsed.count() << std::endl;

    return 0;
}
