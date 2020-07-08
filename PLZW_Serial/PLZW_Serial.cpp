#ifndef STD_LIBS
#define STD_LIBS
#include <stdio.h>
#endif // STD_LIBS
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "lzw.h"
#define IN_PATH "F:\\Dev\\PLZW\\in.txt"

using namespace std;
int main()
{
    string input;
    string line;
    ifstream inFile;
    bool correctness = true;
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
    unsigned int* encodedData = (unsigned int*)malloc(inputLength * sizeof(unsigned int));

    std::chrono::steady_clock::time_point encoding_begin = std::chrono::steady_clock::now();
    const char* input_point = input.c_str();
    unsigned int encodedLength = encoding_lzw(input_point, input.length(), encodedData);
    std::chrono::steady_clock::time_point encoding_end = std::chrono::steady_clock::now();

    encodedData = (unsigned int*)realloc(encodedData, (encodedLength) * sizeof(unsigned int));
    char* decodedData = (char*)malloc(inputLength * sizeof(char));
    //for (unsigned int j = 0; j < encodedLength; j++) {
    //    cout << encodedData[j] << " ";
    //}

    std::chrono::steady_clock::time_point decoding_begin = std::chrono::steady_clock::now();
    unsigned int decodedDataLength = decoding_lzw(encodedData, encodedLength, decodedData);
    std::chrono::steady_clock::time_point decoding_end = std::chrono::steady_clock::now();

    // cout << decodedData << "\n\n";
    
    if (inputLength == decodedDataLength) {
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

    free(encodedData);
    return 0;
}
