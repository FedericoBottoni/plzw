#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "..\PLZW_Serial\lzw.h"
#define IN_PATH "F:\\Dev\\PLZW\\in.txt"
#define DEFAULT_NPROCS 4

unsigned int encoding(int nProcs, string input, unsigned int inputLength, unsigned int* encodedData) {
    const char* input_point = input.c_str();
    unsigned int avgRng, encodedLength = 0, *encodedBuffLengths = new unsigned int[nProcs];
    avgRng = inputLength / nProcs;
    omp_set_num_threads(nProcs);

    #pragma omp parallel shared(nProcs), shared(input_point), shared(inputLength), shared(avgRng), shared(encodedLength), shared(encodedData), shared(encodedBuffLengths), default(none)
    {
        char idProc = omp_get_thread_num();
        unsigned int dataBuffLength, * encodedBuff;
        dataBuffLength = idProc != nProcs - 1 ? avgRng : inputLength - (avgRng * (nProcs - 1));
        encodedBuff = new unsigned int[dataBuffLength];
        const char* shifted_input_point = &input_point[avgRng * idProc];

        encodedBuffLengths[idProc] = encoding_lzw(shifted_input_point, dataBuffLength, encodedBuff);
        #pragma omp barrier

        unsigned int encodedDataOffset = 0;
        for (unsigned int i = 0; i < idProc; i++) {
            encodedDataOffset += encodedBuffLengths[i];
        }

        unsigned int* shifted_encodedData = &encodedData[encodedDataOffset];
        memcpy(shifted_encodedData, encodedBuff, encodedBuffLengths[idProc] * sizeof(unsigned int));

        #pragma omp single
        {
            for (unsigned int i = 0; i < nProcs; i++) {
                encodedLength += encodedBuffLengths[i];
            }
        }
    }
    return encodedLength;
}

using namespace std;
int main(int argc, char* argv[])
{
    string input;
    string line;
    ifstream inFile;
    bool correctness = true;
    int nProcs;

    nProcs = DEFAULT_NPROCS;
    //cout << "Invalid argument '" << argv[1] << "': setting nProc = " << DEFAULT_NPROCS << endl;

    inFile.open(IN_PATH);
    if (!inFile) {
        cout << "Unable to open file";
        exit(1);
    }
    while (inFile >> line) {
        input += line;
    }
    inFile.close();

    unsigned int inputLength = input.length();
    unsigned int* encodedData = new unsigned int[inputLength];
    std::chrono::steady_clock::time_point encoding_begin, encoding_end, decoding_begin, decoding_end;
    encoding_begin = std::chrono::steady_clock::now();

    unsigned int encodedLength = encoding(nProcs, input, inputLength, encodedData);

    encoding_end = std::chrono::steady_clock::now();


    decoding_begin = std::chrono::steady_clock::now();
    string decodedData = decoding_lzw(encodedData, encodedLength);
    decoding_end = std::chrono::steady_clock::now();


    if (inputLength == decodedData.length()) {
        for (unsigned int j = 0; j < inputLength; j++) {
            correctness = input[j] == decodedData[j];
            if (correctness == 0) {
                break;
            }
        }
    }
    else {
        correctness = 0;
    }

    cout << "Lossless propriety: " << correctness;

    cout <<
        "\nChars: " << inputLength << "  Memory: " << inputLength * sizeof(char) << " bytes" <<
        "\nEncoded: " << encodedLength << "  Memory: " << encodedLength * sizeof(unsigned int) << " bytes" << endl;


    cout << "Encoding time: " << std::chrono::duration_cast<std::chrono::milliseconds> (encoding_end - encoding_begin).count() << "[ms]" << std::endl;
    cout << "Decoding time: " << std::chrono::duration_cast<std::chrono::milliseconds> (decoding_end - decoding_begin).count() << "[ms]" << std::endl;

    delete[] encodedData;
    return 0;
}
