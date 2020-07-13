#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
// #include "lzw.h"
#include "../dependencies/uthash.h"
#define IN_PATH "F:\\Dev\\PLZW\\in\\in"
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
    id[length] = '\0';
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
    char* ch = (char*)malloc(sizeof(char));
    for (unsigned int i = 1; i < ALPHABET_LEN; i++) {
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
