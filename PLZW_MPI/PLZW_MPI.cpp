// mpiexec -np 4 PLZW_MPI.exe

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
//#include "..\PLZW_Serial\lzw.h"
#include "../dependencies/uthash.h"
#define IN_PATH "F:\\Dev\\PLZW\\in.txt"
#define ALPHABET_LEN 256



using namespace std;

struct unsorted_node_map {
    char* id; // key
    unsigned int code;
    short tokenSize;
    UT_hash_handle hh; /* makes this structure hashable */
};

struct unsorted_node_map_dec {
    unsigned int id; // key
    char* token;
    short tokenSize;
    UT_hash_handle hh; /* makes this structure hashable */
};

struct unsorted_node_map* table;
struct unsorted_node_map_dec* table_dec;

void push_into_table(char* id, short tokenSize, unsigned int code) {
    struct unsorted_node_map* s = (struct unsorted_node_map*)malloc(sizeof * s);
    char* curr_token = (char*)malloc((tokenSize + 1) * sizeof(char));
    memcpy(curr_token, id, tokenSize);
    curr_token[tokenSize] = '\0';
    s->id = curr_token;
    s->tokenSize = tokenSize;
    s->code = code;
    HASH_ADD_KEYPTR(hh, table, s->id, tokenSize, s);
}

struct unsorted_node_map* find_by_token(char* id, short length) {
    struct unsorted_node_map* s;
    HASH_FIND_STR(table, id, s);
    return s;
}

struct unsorted_node_map* find_by_code(unsigned int code) {
    struct unsorted_node_map* node, * tmp;
    HASH_ITER(hh, table, node, tmp) {
        if (node->code == code) {
            return node;
        }
    }
    return NULL;
}

void dispose_table() {
    struct unsorted_node_map* node, * tmp;

    HASH_ITER(hh, table, node, tmp) {
        free(node->id);
        HASH_DEL(table, node);
        free(node);
    }
}

void push_into_table_dec(unsigned int id, char* token, short tokenSize) {
    struct unsorted_node_map_dec* s = (struct unsorted_node_map_dec*)malloc(sizeof * s);
    char* curr_token = (char*)malloc((tokenSize + 1) * sizeof(char));
    memcpy(curr_token, token, tokenSize);
    curr_token[tokenSize] = '\0';
    s->token = curr_token;
    s->tokenSize = tokenSize;
    s->id = id;
    HASH_ADD_INT(table_dec, id, s);
}

struct unsorted_node_map_dec* find_by_code_dec(unsigned int id) {
    struct unsorted_node_map_dec* s;
    HASH_FIND_INT(table_dec, &id, s);
    return s;
}

void dispose_table_dec() {
    struct unsorted_node_map_dec* node, * tmp;

    HASH_ITER(hh, table_dec, node, tmp) {
        free(node->token);
        HASH_DEL(table_dec, node);
        free(node);
    }
}

int encoding_lzw(const char* s1, unsigned int count, unsigned int* objectCode)
{
    table = NULL;
    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch[0] = char(i);
        push_into_table(ch, 1, i);
    }
    free(ch);

    unsorted_node_map* node;
    int out_index = 0, pLength;
    char* p = (char*)malloc(2 * sizeof(char)), * pandc = (char*)malloc(3 * sizeof(char)), * c = new char[1];
    p[0] = s1[0];
    p[1] = '\0';
    pandc[2] = '\0';
    pLength = 1;
    unsigned int code = ALPHABET_LEN;
    unsigned int i;
    for (i = 0; i < count; i++) {
        if (i != count - 1)
            c[0] = s1[i + 1];
        for (unsigned short str_i = 0; str_i < pLength; str_i++) pandc[str_i] = p[str_i];
        pandc[pLength] = c[0];
        unsorted_node_map* node = find_by_token(pandc, pLength + 1);
        if (node != NULL) {
            p = (char*)realloc(p, (++pLength + 1) * sizeof(char));
            pandc = (char*)realloc(pandc, (pLength + 2) * sizeof(char));
            p[pLength] = '\0';
            pandc[pLength + 1] = '\0';
            for (unsigned short str_i = 0; str_i < pLength; str_i++) p[str_i] = pandc[str_i];
        }
        else {
            node = find_by_token(p, pLength);
            objectCode[out_index++] = node->code;
            push_into_table(pandc, pLength + 1, code);
            code++;
            p[0] = c[0];
            if (pLength > 1) {
                p = (char*)realloc(p, 2 * sizeof(char));
                pandc = (char*)realloc(pandc, 3 * sizeof(char));
                p[1] = '\0';
                pandc[2] = '\0';
            }
            pLength = 1;
        }
        c[0] = NULL;
    }
    objectCode[out_index++] = find_by_token(p, pLength)->code;

    free(p);
    free(pandc);
    dispose_table();
    return out_index;
}

unsigned int decoding_lzw(unsigned int* op, int op_length, char* decodedData)
{
    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 1; i < ALPHABET_LEN; i++) {
        ch[0] = char(i);
        push_into_table_dec(i, ch, 1);
    }
    free(ch);

    unsigned int old = op[0], decodedDataLength, n;
    struct unsorted_node_map_dec* temp_node, * s_node = find_by_code_dec(old);
    int temp_length = 0, s_length = s_node->tokenSize;
    char* s = (char*)malloc((s_length + 1) * sizeof(char)), * temp = (char*)malloc(sizeof(char));
    memcpy(s, s_node->token, s_length);
    s[s_length] = '\0';
    temp[0] = '\0';
    char c = s[0];

    memcpy(decodedData, s, s_length);
    decodedDataLength = 1;
    int count = ALPHABET_LEN;
    for (int i = 0; i < op_length - 1; i++) {
        n = op[i + 1];
        if (find_by_code_dec(n) == NULL) {
            s_node = find_by_code_dec(old);
            s_length = s_node->tokenSize;
            s = (char*)realloc(s, (++s_length + 1) * sizeof(char));
            memcpy(s, s_node->token, s_length - 1);
            s[s_length - 1] = c;
            s[s_length] = '\0';
        }
        else {
            s_node = find_by_code_dec(n);
            if (s_node->tokenSize != s_length) {
                s_length = s_node->tokenSize;
                s = (char*)realloc(s, (s_length + 1) * sizeof(char));
                s[s_length] = '\0';
            }
            memcpy(s, s_node->token, s_length);
        }
        memcpy(&decodedData[decodedDataLength], s, s_length);
        decodedDataLength += s_length;
        c = s[0];
        temp_node = find_by_code_dec(old);
        if (temp_length != temp_node->tokenSize + 1) {
            temp_length = temp_node->tokenSize + 1;
            temp = (char*)realloc(temp, (temp_length + 1) * sizeof(char));
            temp[temp_length] = '\0';
        }
        memcpy(temp, temp_node->token, temp_length - 1);
        temp[temp_length - 1] = c;
        push_into_table_dec(count, temp, temp_length);
        count++;
        old = n;
    }
    free(temp);
    free(s);
    dispose_table_dec();
    return decodedDataLength;
}

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

unsigned int decoding(unsigned int* encodedData, int encodedLength, int* encStartPos, int rank, int size, int root, char* decodedData, unsigned int decodedExpectedLength) {
    int decodedLengthBuff, encodedLengthBuff, * decodedLengthBuffs = new int[size], *counts = new int[size];
    unsigned int decodedLength, dataBuffLength, avgRng = decodedExpectedLength / size;
    dataBuffLength = rank != size - 1 ? avgRng : decodedExpectedLength - (avgRng * (size - 1));
    char* decodedBuff = (char*)malloc(dataBuffLength * sizeof(char));
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

    decodedLengthBuff = decoding_lzw(encodedDataBuff, counts[rank], decodedBuff);

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
        decodedBuff,
        decodedLengthBuff,
        MPI_CHAR,
        decodedData,
        decodedLengthBuffs,
        encStartPos,
        MPI_CHAR,
        root,
        MPI_COMM_WORLD);

    free(decodedBuff);
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
    int decodedLength = decoding(encodedData, encodedLength, encStartPos, rank, size, root, decodedData, inputLength);
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

