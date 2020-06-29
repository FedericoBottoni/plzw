// mpiexec -np 4 PLZW_MPI.exe

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "..\PLZW_Serial\lzw.h"
#define IN_PATH "F:\\Dev\\PLZW\\in.txt"

using namespace std;
unsigned int encoding(string input, int inputLength, int rank, int size, int root, unsigned int* encodedData, int* encStartPos) {

    unsigned int avgRng, receiveCount;
    unsigned int* startPoint;
    int* counts;
    char* inputBuf;

    startPoint = new unsigned int[size];
    counts = new int[size];

    avgRng = inputLength / size;
    receiveCount = avgRng;
    if (rank == size - 1) {
        receiveCount = inputLength - (size - 1) * avgRng;
    }
    inputBuf = new char[receiveCount];

    for (unsigned short p = 0; p < size; p++) {
        startPoint[p] = avgRng * p;
        counts[p] = avgRng;
        encStartPos[p] = avgRng * p;
    }
    counts[size - 1] = inputLength - (size - 1) * avgRng;
    MPI_Scatterv(
        input.c_str(),
        counts,
        encStartPos,
        MPI_CHAR,
        inputBuf,
        receiveCount,
        MPI_CHAR,
        root,
        MPI_COMM_WORLD);

    unsigned int* encodedDataBuff = new unsigned int[receiveCount];
    int* encodedLengthBuffs = new int[size];

    unsigned int encodedLengthBuff = encoding_lzw(inputBuf, receiveCount, encodedDataBuff);

    for (unsigned short p = 0; p < size; p++) {
        counts[p] = 1;
        encStartPos[p] = p;
    }

    MPI_Allgather(
        &encodedLengthBuff,
        1,
        MPI_INT,
        encodedLengthBuffs,
        1,
        MPI_INT,
        MPI_COMM_WORLD);

    unsigned int encodedLength = 0;
    for (unsigned short p = 0; p < size; p++) {
        encStartPos[p] = encodedLength;
        encodedLength += encodedLengthBuffs[p];
    }

    MPI_Gatherv(
        encodedDataBuff,
        encodedLengthBuff,
        MPI_INT,
        encodedData,
        encodedLengthBuffs,
        encStartPos,
        MPI_INT,
        root,
        MPI_COMM_WORLD);

    delete[] encodedDataBuff; 
    delete[] encodedLengthBuffs;
    delete[] startPoint;
    delete[] counts; 
    delete[] inputBuf;

    return encodedLength;
}

unsigned int decoding(unsigned int* encodedData, int encodedLength, int* encStartPos, int rank, int size, int root, char* decodedData) {
    unsigned int decodedLength;
    int decodedLengthBuff, encodedLengthBuff;
    int* decodedLengthBuffs = new int[size];

    int* counts = new int[size];

    for (unsigned short p = 0; p < size - 1; p++) {
        counts[p] = encStartPos[p + 1] - encStartPos[p];
    }
    counts[size - 1] = encodedLength - encStartPos[size - 1];

    encodedLengthBuff = counts[rank];
    unsigned int* encodedDataBuff = new unsigned int[encodedLengthBuff];

    MPI_Scatterv(
        encodedData,
        counts,
        encStartPos,
        MPI_INT,
        encodedDataBuff,
        counts[rank],
        MPI_INT,
        root,
        MPI_COMM_WORLD);

    string decodedDataBuff = decoding_lzw(encodedDataBuff, counts[rank]);

    decodedLengthBuff = decodedDataBuff.length();
    MPI_Allgather(
        &decodedLengthBuff,
        1,
        MPI_INT,
        decodedLengthBuffs,
        1,
        MPI_INT,
        MPI_COMM_WORLD);


    decodedLength = 0;
    for (unsigned short p = 0; p < size; p++) {
        encStartPos[p] = decodedLength;
        decodedLength += decodedLengthBuffs[p];
    }

    MPI_Gatherv(
        decodedDataBuff.c_str(),
        decodedLengthBuff,
        MPI_CHAR,
        decodedData,
        decodedLengthBuffs,
        encStartPos,
        MPI_CHAR,
        root,
        MPI_COMM_WORLD);


    delete[] decodedLengthBuffs;
    delete[] encodedDataBuff;
    delete[] counts;
    return decodedLength;
}

int main()
{
    int size, rank;
    int root = 0;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    string input;
    unsigned int inputLength;
    std::chrono::steady_clock::time_point encoding_begin, encoding_end, decoding_begin, decoding_end;
    bool correctness = true;
    if (rank == root) {
        ifstream inFile;
        inFile.open(IN_PATH);
        if (!inFile) {
            cout << "Unable to open file";
            exit(1);
        }
        string line;
        while (inFile >> line) {
            input += line;
        }
        line.clear();
        inFile.close();

        inputLength = input.length();
        encoding_begin = std::chrono::steady_clock::now();
    }

    MPI_Bcast(
        &inputLength,
        1,
        MPI_INT,
        root,
        MPI_COMM_WORLD
    );

    unsigned int* encodedData = new unsigned int[inputLength];
    int* encStartPos = new int[size];
    int encodedLength = encoding(input, inputLength, rank, size, root, encodedData, encStartPos);

    if (rank == root) {
        encoding_end = std::chrono::steady_clock::now();
    }
    /** END ENCODING */


    /** BEGIN DECODING: params: "encodedData" and "encStartPos" */
    if (rank == root) {
        decoding_begin = std::chrono::steady_clock::now();
    }
    char* decodedData = new char[inputLength];
    int decodedLength = decoding(encodedData, encodedLength, encStartPos, rank, size, root, decodedData);
    if (rank == root) {
        decoding_end = std::chrono::steady_clock::now();
    }

    if (rank == root) {
        if (inputLength == decodedLength) {
            const char* input_point = input.c_str(); 
            for (int i = 0; i < input.length(); i++) {
                if (input_point[i] != decodedData[i]) {
                    correctness = 0;
                    cout << "Found mistake in position i=" << i << " input='" << input[i] << "' and decoded='" << decodedData[i] << "'" << endl;
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

    }

    delete[] encStartPos;
    delete[] encodedData;
    MPI_Barrier(MPI_COMM_WORLD);
    delete[] decodedData;

    MPI_Finalize();
    return 0;
}

