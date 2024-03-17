#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "progress_bar.hpp"

using namespace std;

const unsigned char check = 0b10000000;

struct TranslationNode
{
    TranslationNode *zero, *one;
    unsigned char character;
};

progress PROGRESS;

long int read_file_size(unsigned char &, int, FILE *);
void write_file_name(char *, int, unsigned char &, int &, TranslationNode *, FILE *);
void translate_file(char *, long int, unsigned char &, int &, TranslationNode *, FILE *);

unsigned char process_8_bits_NUMBER(unsigned char &, int, FILE *);
void process_n_bits_TO_STRING(unsigned char &, int, int &, FILE *, TranslationNode *, unsigned char);

bool file_exists(char *);
void change_name_if_exists(char *);

void burn_tree(TranslationNode *);

/*          CONTENT TABLE IN ORDER
    compressed file's composition is in order below
    that is why we re going to translate it part by part

.first (one byte)           ->  letter_count
.third (bit groups)
    3.1 (8 bits)            ->  current unique byte
    3.2 (8 bits)            ->  length of the transformation
    4.3 (bits)              ->  transformation code of that unique byte
    .sixth (8 bytes)        ->  size of current file (IF FILE)
    .eighth (a lot of bits) ->  translate and write current file (IF FILE)

*whenever we see a new folder we will write seventh then
    start writing the files(and folders) inside the current folder from fourth to eighth
**groups from fifth to eighth will be written as much as the file count
*/

int main(int argc, char *argv[])
{
    int uniqueByteCount = 0;
    FILE *compressedFile, *newFile;
    if (argc != 2)
    {
        std::cout << "Missing compressed file name." << endl
                  << "Usage: './extract <compressed_file_name>'" << endl;
        return 1;
    }

    compressedFile = fopen(argv[1], "rb");
    if (!compressedFile)
    {
        std::cout << argv[1] << " does not exist" << endl;
        return 0;
    }

    // Initialize the progress bar with the total size of the compressed file.
    fseek(compressedFile, 0, SEEK_END);
    PROGRESS.MAX = ftell(compressedFile); // Setting progress bar maximum value
    fseek(compressedFile, 0, SEEK_SET);

    // Reading the unique byte count from the compressed file's header.
    fread(&uniqueByteCount, 1, 1, compressedFile);
    if (uniqueByteCount == 0)
        uniqueByteCount = 256; // Handling the special case where there are 256 unique bytes.

    unsigned char bufferByte = 0, decodedCharacter;
    int bitCounter = 0, transformationLength;
    TranslationNode *root = new TranslationNode;
    root->zero = nullptr;
    root->one = nullptr;

    // Reading transformation information for each unique byte and storing it in the translation tree.
    for (int i = 0; i < uniqueByteCount; i++)
    {
        decodedCharacter = process_8_bits_NUMBER(bufferByte, bitCounter, compressedFile);
        transformationLength = process_8_bits_NUMBER(bufferByte, bitCounter, compressedFile);
        if (transformationLength == 0)
            transformationLength = 256;
        process_n_bits_TO_STRING(bufferByte, transformationLength, bitCounter, compressedFile, root, decodedCharacter);
    }

    // File count was written to the compressed file from least significiant byte
    // to most significiant byte to make sure system's endianness
    // does not affect the process and that is why we are processing size information like this

    long int fileSize = read_file_size(bufferByte, bitCounter, compressedFile);
    int fileNameLength = process_8_bits_NUMBER(bufferByte, bitCounter, compressedFile);
    char newfileName[fileNameLength + 4];
    write_file_name(newfileName, fileNameLength, bufferByte, bitCounter, root, compressedFile);
    change_name_if_exists(newfileName);

    // Translating and writing the file content based on the Huffman encoding.
    translate_file(newfileName, fileSize, bufferByte, bitCounter, root, compressedFile);

    // Clean up.
    fclose(compressedFile);
    burn_tree(root);
    std::cout << "Decompression is complete" << endl;
}

// burn_tree function is used for deallocating translation tree
void burn_tree(TranslationNode *node)
{
    if (node->zero)
        burn_tree(node->zero);
    if (node->one)
        burn_tree(node->one);
    delete node;
}

// process_n_bits_TO_STRING function reads n successive bits from the compressed file
// and stores it in a leaf of the translation tree,
// after creating that leaf and sometimes after creating nodes that are binding that leaf to the tree.
void process_n_bits_TO_STRING(unsigned char &current_byte, int n, int &current_bit_count, FILE *fp_read, TranslationNode *node, unsigned char uChar)
{
    for (int i = 0; i < n; i++)
    {
        if (current_bit_count == 0)
        {
            fread(&current_byte, 1, 1, fp_read);
            current_bit_count = 8;
        }

        switch (current_byte & check)
        {
        case 0:
            if (!(node->zero))
            {
                node->zero = new TranslationNode;
                node->zero->zero = NULL;
                node->zero->one = NULL;
            }
            node = node->zero;
            break;
        case 128:
            if (!(node->one))
            {
                node->one = new TranslationNode;
                node->one->zero = NULL;
                node->one->one = NULL;
            }
            node = node->one;
            break;
        }
        current_byte <<= 1;
        current_bit_count--;
    }
    node->character = uChar;
}

// process_8_bits_NUMBER reads 8 successive bits from compressed file
//(does not have to be in the same byte)
// and returns it in unsigned char form
unsigned char process_8_bits_NUMBER(unsigned char &current_byte, int current_bit_count, FILE *fp_read)
{
    unsigned char val, temp_byte;
    fread(&temp_byte, 1, 1, fp_read);
    val = current_byte | (temp_byte >> current_bit_count);
    current_byte = temp_byte << 8 - current_bit_count;
    return val;
}

void change_name_if_exists(char *name)
{
    char *i;
    int copy_count;
    if (file_exists(name))
    {
        char *dot_pointer = NULL;
        for (i = name; *i; i++)
        {
            if (*i == '.')
                dot_pointer = i;
        }
        if (dot_pointer)
        {
            string s = dot_pointer;
            strcpy(dot_pointer, "(1)");
            dot_pointer++;
            strcpy(dot_pointer + 2, &s[0]);
        }
        else
        {
            dot_pointer = i;
            strcpy(dot_pointer, "(1)");
            dot_pointer++;
        }
        for (copy_count = 1; copy_count < 10; copy_count++)
        {
            *dot_pointer = (char)('0' + copy_count);
            if (!file_exists(name))
            {
                break;
            }
        }
    }
}

// checks if the file or folder exists
bool file_exists(char *name)
{
    FILE *fp = fopen(name, "rb");
    if (fp)
    {
        fclose(fp);
        return 1;
    }
    else
    {
        DIR *dir = opendir(name);
        if (dir)
        {
            closedir(dir);
            return 1;
        }
    }
    return 0;
}

// returns file's size
long int read_file_size(unsigned char &current_byte, int current_bit_count, FILE *fp_compressed)
{
    long int size = 0;
    {
        long int multiplier = 1;
        for (int i = 0; i < 8; i++)
        {
            size += process_8_bits_NUMBER(current_byte, current_bit_count, fp_compressed) * multiplier;
            multiplier *= 256;
        }
    }
    PROGRESS.current(ftell(fp_compressed)); // updating progress bar
    return size;
    // Size was written to the compressed file from least significiant byte
    // to the most significiant byte to make sure system's endianness
    // does not affect the process and that is why we are processing size information like this
}

// Decodes current file's name and writes file name to newfile char array
void write_file_name(char *newfile, int file_name_length, unsigned char &current_byte, int &current_bit_count, TranslationNode *root, FILE *fp_compressed)
{
    TranslationNode *node;
    newfile[file_name_length] = 0;
    for (int i = 0; i < file_name_length; i++)
    {
        node = root;
        while (node->zero || node->one)
        {
            if (current_bit_count == 0)
            {
                fread(&current_byte, 1, 1, fp_compressed);
                current_bit_count = 8;
            }
            if (current_byte & check)
            {
                node = node->one;
            }
            else
            {
                node = node->zero;
            }
            current_byte <<= 1;
            current_bit_count--;
        }
        newfile[i] = node->character;
    }
}

// This function translates compressed file from info that is now stored in the translation tree
// then writes it to a newly created file
void translate_file(char *path, long int size, unsigned char &current_byte, int &current_bit_count, TranslationNode *root, FILE *fp_compressed)
{
    TranslationNode *node;
    FILE *fp_new = fopen(path, "wb");
    for (long int i = 0; i < size; i++)
    {
        node = root;
        while (node->zero || node->one)
        {
            if (current_bit_count == 0)
            {
                fread(&current_byte, 1, 1, fp_compressed);
                current_bit_count = 8;
            }
            if (current_byte & check)
            {
                node = node->one;
            }
            else
            {
                node = node->zero;
            }
            current_byte <<= 1;
            current_bit_count--;
        }
        fwrite(&(node->character), 1, 1, fp_new);
    }
    fclose(fp_new);
}