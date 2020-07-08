#ifndef LZH
#define LZH
#include "../dependencies/uthash.h"
#define ALPHABET_LEN 256
#define MAX_TOKEN_SIZE 1000
#ifndef STD_LIBS
#define STD_LIBS
#include <stdio.h>
#endif // STD_LIBS


using namespace std;

struct unsorted_node_map {
    char* id; // token
    unsigned int code;
    short tokenSize;
    UT_hash_handle hh; /* makes this structure hashable */
};

struct unsorted_node_map* table = NULL;

void push_into_table(char* id, short tokenSize, unsigned int code) {
    struct unsorted_node_map* s;
    s->id = id;
    s->tokenSize = tokenSize;
    s->code = code;
    HASH_ADD_STR(table, id, s);
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
    struct unsorted_node_map* node, *tmp;

    HASH_ITER(hh, table, node, tmp) {
        HASH_DEL(table, node);
        free(node);
    }
}

int encoding_lzw(const char* s1, unsigned int count, unsigned int* objectCode)
{
    char ch;
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch = char(i);
        push_into_table(&ch, 1, i);
    }

    int out_index = 0, pLength;
    char* p = new char[MAX_TOKEN_SIZE], * pandc = new char[MAX_TOKEN_SIZE], * c = new char[1];
    p[0] = s1[0];
    pLength = 1;
    unsigned int code = ALPHABET_LEN;
    unsigned int i;
    for (i = 0; i < count; i++) {
        if (i != count - 1)
            c[0] = s1[i + 1];
        for (unsigned short str_i = 0; str_i < pLength; str_i++) pandc[str_i] = p[str_i];
        pandc[pLength] = c[0];
        if (find_by_token(pandc, pLength + 1) != NULL) {
            pLength++;
            for (unsigned short str_i = 0; str_i < pLength; str_i++) p[str_i] = pandc[str_i];
        }
        else {
            objectCode[out_index++] = find_by_token(p, pLength)->code;
            push_into_table(pandc, pLength + 1, code);
            code++;
            memset(p, 0, sizeof(p));
            p[0] = c[0];
            pLength = 1;
        }
        c[0] = NULL;
        //memset(pandc, 0, sizeof(pandc));
        if (pLength > MAX_TOKEN_SIZE - 1) {
            printf("Token-size is not enough big\n");
            exit(-1);
        }
    }
    objectCode[out_index] = find_by_token(p, pLength)->code;

    delete[] p;
    delete[] pandc;
    dispose_table();
    return out_index;
}

unsigned int decoding_lzw(unsigned int* op, int op_length, char* decodedData)
{
    char ch;
    for (unsigned int i = 0; i < ALPHABET_LEN; i++) {
        ch = char(i);
        push_into_table(&ch, 1, i);
    }

    unsigned int old = op[0], decodedDataLength, n;
    struct unsorted_node_map* temp_node, * s_node = find_by_code(old);
    int temp_length, s_length = s_node->tokenSize;
    char* s = new char[MAX_TOKEN_SIZE], * temp = new char[MAX_TOKEN_SIZE];
    memcpy(s, s_node->id, s_length);
    char* c = s;
    memcpy(decodedData, s, s_length);
    decodedDataLength = 1;
    int count = ALPHABET_LEN;
    for (int i = 0; i < op_length - 1; i++) {
        n = op[i + 1];
        if (find_by_token(s, s_length) == NULL) {
            s_node = find_by_code(old);
            s_length = s_node->tokenSize;
            memcpy(s, s_node->id, s_length);
            s[s_length++] = *c;
        }
        else {
            s_node = find_by_code(n);
            s_length = s_node->tokenSize;
            memcpy(s, s_node->id, s_length);
        }
        if (s_length > MAX_TOKEN_SIZE - 1) {
            printf("Token-size is not enough big");
            exit(-1);
        }
        memcpy(&decodedData[decodedDataLength], s, s_length);
        decodedDataLength += s_length;
        c = s;
        temp_node = find_by_code(old);
        temp_length = temp_node->tokenSize;
        memcpy(temp, temp_node->id, temp_length);
        temp[temp_length] = *c;
        push_into_table(temp, temp_length + 1, count);
        count++;
        old = n;
    }
    delete[] temp;
    delete[] s;
    dispose_table();
    return decodedDataLength;
}

#endif // LZW