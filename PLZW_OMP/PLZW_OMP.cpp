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

unsigned int encoding(int nProcs, string input, unsigned int inputLength, unsigned int* encodedData, unsigned int* encStartPos) {
    const char* input_point = input.c_str();
    unsigned int avgRng, encodedLength = 0, *encodedBuffLengths = new unsigned int[nProcs];
    avgRng = inputLength / nProcs;
    omp_set_num_threads(nProcs);

    #pragma omp parallel shared(nProcs), shared(input_point), shared(inputLength), shared(avgRng), shared(encodedLength), shared(encodedData), shared(encodedBuffLengths), shared(encStartPos), default(none)
    {
        char idProc = omp_get_thread_num();
        unsigned int dataBuffLength, * encodedBuff;
        dataBuffLength = idProc != nProcs - 1 ? avgRng : inputLength - (avgRng * (nProcs - 1));
        encodedBuff = new unsigned int[dataBuffLength];
        const char* shifted_input_point = &input_point[avgRng * idProc];

        encodedBuffLengths[idProc] = encoding_lzw(shifted_input_point, dataBuffLength, encodedBuff);
        #pragma omp barrier

        unsigned int encodedDataOffset = 0;
        for (unsigned short i = 0; i < idProc; i++) {
            encodedDataOffset += encodedBuffLengths[i];
        }

        unsigned int* shifted_encodedData = &encodedData[encodedDataOffset];
        memcpy(shifted_encodedData, encodedBuff, encodedBuffLengths[idProc] * sizeof(unsigned int));

        encStartPos[idProc] = encodedDataOffset;
        #pragma omp single
        {
            for (unsigned short i = 0; i < nProcs; i++) {
                encodedLength += encodedBuffLengths[i];
            }
        }

        delete[] encodedBuff;
        //delete shifted_input_point;
        //delete shifted_encodedData;
    }
    return encodedLength;
}


unsigned int decoding(int nProcs, unsigned int* encodedData, unsigned int encodedLength, unsigned int* encStartPos, char* decodedData) {
    unsigned int decodedLength = 0;
    int* decodedBuffLengths = new int[nProcs];
    #pragma omp parallel shared(nProcs), shared(encodedData), shared(encodedLength), shared(encStartPos), shared(decodedData), shared(decodedBuffLengths), shared(decodedLength), default(none)
    {
        char idProc = omp_get_thread_num();
        int encodedBuffLength, * encodedBuffLengths = new int[nProcs];
        for (unsigned short p = 0; p < nProcs - 1; p++) {
            encodedBuffLengths[p] = encStartPos[p + 1] - encStartPos[p];
        }
        encodedBuffLengths[nProcs - 1] = encodedLength - encStartPos[nProcs - 1];
        encodedBuffLength = encodedBuffLengths[idProc];

        unsigned int encodedDataOffset = 0;
        for (unsigned short i = 0; i < idProc; i++) {
            encodedDataOffset += encodedBuffLengths[i];
        }
        unsigned int* shifted_encodedData = &encodedData[encodedDataOffset];

        string decodedDataBuff = decoding_lzw(shifted_encodedData, encodedBuffLength);
        decodedBuffLengths[idProc] = decodedDataBuff.length();
        #pragma omp barrier

        unsigned int decodedDataOffset = 0;
        for (unsigned short i = 0; i < idProc; i++) {
            decodedDataOffset += decodedBuffLengths[i];
        }

        char* shifted_decodedData = &decodedData[decodedDataOffset];
        memcpy(shifted_decodedData, decodedDataBuff.c_str(), decodedBuffLengths[idProc] * sizeof(char));

        #pragma omp single
        {
            for (unsigned short i = 0; i < nProcs; i++) {
                decodedLength += decodedBuffLengths[i];
            }
        }

        delete[] encodedBuffLengths;
        //delete shifted_encodedData;
    }
    return decodedLength;
}

using namespace std;
int main()
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
    unsigned int* encStartPos = new unsigned int[nProcs];
    unsigned int encodedLength = encoding(nProcs, input, inputLength, encodedData, encStartPos);
    encoding_end = std::chrono::steady_clock::now();


    decoding_begin = std::chrono::steady_clock::now();
    char* decodedData = new char[inputLength];
    int decodedLength = decoding(nProcs, encodedData, encodedLength, encStartPos, decodedData);
    decoding_end = std::chrono::steady_clock::now();


    if (inputLength == decodedLength) {
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
    delete[] encStartPos;
    delete[] decodedData;
    return 0;
}
