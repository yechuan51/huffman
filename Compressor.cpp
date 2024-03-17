#include <iostream>
#include <cstdio>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include "progress_bar.hpp"

using namespace std;

void write_from_uChar(unsigned char, unsigned char &, int, FILE *);

int this_is_not_a_folder(char *);
long int size_of_the_file(char *);
void count_in_folder(string, long int *, long int &, long int &);

void write_file_count(int, unsigned char &, int, FILE *);
void write_file_size(long int, unsigned char &, int, FILE *);
void write_file_name(char *, string *, unsigned char &, int &, FILE *);
void write_the_file_content(FILE *, long int, string *, unsigned char &, int &, FILE *);
void write_the_folder(string, string *, unsigned char &, int &, FILE *);

/*          CONTENT TABLE IN ORDER
---------PART 1-CALCULATING TRANSLATION INFO----------
Important Note:4 and 5 are the most important parts of this algorithm
If you dont know how Huffman's algorithm works I really recommend you to check this link before you continue:
https://en.wikipedia.org/wiki/Huffman_coding#Basic_technique

1-Size information
2-Counting usage frequency of unique bytes and unique byte count
3-Creating the base of the translation array
4-Creating the translation tree inside the translation array by weight distribution
5-adding strings from top to bottom to create translated versions of unique bytes

---------PART 2-CREATION OF COMPRESSED FILE-----------
    Compressed File's structure had been documented below

first (one byte)            ->  letter_count
second (bit group)
    2.1 (one byte)          ->  password_length
    2.2 (bytes)             ->  password (if password exists)
third (bit groups)
    3.1 (8 bits)            ->  current unique byte
    3.2 (8 bits)            ->  length of the transformation
    3.3 (bits)              ->  transformation code of that unique byte

fourth (2 bytes)**          ->  file_count (inside the current folder)
    fifth (1 bit)*          ->  file or folder information  ->  folder(0) file(1)
    sixth (8 bytes)         ->  size of current input_file (IF FILE)
    seventh (bit group)
        7.1 (8 bits)        ->  length of current input_file's or folder's name
        7.2 (bits)          ->  transformed version of current input_file's or folder's name
    eighth (a lot of bits)  ->  transformed version of current input_file (IF FILE)

*whenever we see a new folder we will write seventh then start writing from fourth to eighth
**groups from fifth to eighth will be written as much as file count in that folder
    (this is argument_count-1(argc-1) for the main folder)

*/

progress PROGRESS;

struct TreeNode
{ // this structure will be used to create the translation tree
    TreeNode *left, *right;
    long int occurances;
    unsigned char character;
    string bit;
};

bool TreeNodeCompare(TreeNode a, TreeNode b)
{
    return a.occurances < b.occurances;
}

int main(int argc, char *argv[])
{
    long int occurances[256];
    long int totalBitCount = 0;
    int letter_count = 0;
    if (argc == 1)
    {
        std::cout << "Missing file name" << endl
                  << "try './archive {{file_name}}'" << endl;
        return 0;
    }
    for (long int *i = occurances; i < occurances + 256; i++)
    {
        *i = 0;
    }

    FILE *originalFilePtr;

    for (int i = 1; i < argc; i++)
    { // checking for wrong input
        if (!this_is_not_a_folder(argv[i]))
        {
            std::cout << "You cannot compress a directory" << endl;
            return 0;
        }
        originalFilePtr = fopen(argv[i], "rb");
        if (!originalFilePtr)
        {
            std::cout << argv[i] << " file does not exist" << endl
                      << "Process has been terminated" << endl;
            return 0;
        }
        fclose(originalFilePtr);
    }

    // Histograming the frequency of bytes.
    unsigned char *x_ptr, x; // these are temp variables to take input from the file
    x_ptr = &x;
    long int totalSize = 0, size;
    totalBitCount += 16 + 9 * (argc - 1);
    for (int current_file = 1; current_file < argc; current_file++)
    {
        for (char *c = argv[current_file]; *c; c++)
        { // counting usage frequency of unique bytes on the file name (or folder name)
            occurances[(unsigned char)(*c)]++;
        }

        size = size_of_the_file(argv[current_file]);
        totalSize += size;
        totalBitCount += 64;

        // "rb" is for reading binary files
        originalFilePtr = fopen(argv[current_file], "rb");
        // reading the first byte of the file into x.
        fread(x_ptr, 1, 1, originalFilePtr);
        for (long int i = 0; i < size; i++)
        { // counting usage frequency of unique bytes inside the file
            occurances[x]++;
            fread(x_ptr, 1, 1, originalFilePtr);
        }
        fclose(originalFilePtr);
    }

    // Traverse through all possible bytes and count the number of unique bytes.
    for (long int *i = occurances; i < occurances + 256; i++)
    {
        if (*i)
        {
            letter_count++;
        }
    }
    //---------------------------------------------

    //--------------------3------------------------
    // Step 1: Initialize the leaf nodes for Huffman tree construction.
    // Each leaf node represents a unique byte and its frequency in the input data.
    TreeNode nodesForHuffmanTree[letter_count * 2 - 1];
    TreeNode *currentNode = nodesForHuffmanTree;

    // Step 2: Fill the array with data for each unique byte.
    for (long int *frequency = occurances; frequency < occurances + 256; frequency++)
    {
        if (*frequency)
        {
            currentNode->right = NULL;
            currentNode->left = NULL;
            currentNode->occurances = *frequency;
            currentNode->character = frequency - occurances;
            currentNode++;
        }
    }

    // Step 3: Sort the leaf nodes based on frequency to prepare for tree construction.
    // In ascending order.
    sort(nodesForHuffmanTree, nodesForHuffmanTree + letter_count, TreeNodeCompare);

    // Step 4: Construct the Huffman tree by merging nodes with the lowest frequencies.
    TreeNode *smallestNode = nodesForHuffmanTree;
    TreeNode *secondSmallestNode = nodesForHuffmanTree + 1;
    TreeNode *newInternalNode = nodesForHuffmanTree + letter_count;
    TreeNode *nextInternalNode = nodesForHuffmanTree + letter_count;
    TreeNode *nextLeafNode = nodesForHuffmanTree + 2;
    for (int i = 0; i < letter_count - 1; i++)
    {
        // Create a new internal node that combines the two smallest nodes.
        newInternalNode->occurances = smallestNode->occurances + secondSmallestNode->occurances;
        newInternalNode->left = smallestNode;
        newInternalNode->right = secondSmallestNode;
        // Assign bits for tree navigation: '1' for the path to smallestNode,
        // '0' for secondSmallestNode.
        smallestNode->bit = "1";
        secondSmallestNode->bit = "0";
        newInternalNode++;

        // Update smallestNode and secondSmallestNode for the next iteration.
        if (nextLeafNode >= nodesForHuffmanTree + letter_count)
        {
            // All leaf nodes have been processed; proceed with internal nodes.
            smallestNode = nextInternalNode;
            nextInternalNode++;
        }
        else
        {
            // Choose the next smallest node from the leaf or internal nodes.
            smallestNode = (nextLeafNode->occurances < nextInternalNode->occurances) ? nextLeafNode++ : nextInternalNode++;
        }

        // Repeat the process for secondSmallestNode.
        if (nextLeafNode >= nodesForHuffmanTree + letter_count)
        {
            secondSmallestNode = nextInternalNode;
            nextInternalNode++;
        }
        else if (nextInternalNode >= newInternalNode)
        {
            secondSmallestNode = nextLeafNode;
            nextLeafNode++;
        }
        else
        {
            secondSmallestNode = (nextLeafNode->occurances < nextInternalNode->occurances) ? nextLeafNode++ : nextInternalNode++;
        }
    }

    // Step 5: Assign Huffman codes to each node.
    // Iterate from the last internal node to the root, building the Huffman codes in reverse.
    for (TreeNode *node = nodesForHuffmanTree + letter_count * 2 - 2; node > nodesForHuffmanTree - 1; node--)
    {
        // If a left child exists, concatenate the current node's code to it. This assigns the '0' path.
        if (node->left)
        {
            node->left->bit = node->bit + node->left->bit;
        }

        // Similar operation for the right child, representing the '1' path.
        if (node->right)
        {
            node->right->bit = node->bit + node->right->bit;
        }
    }

    string scompressed = argv[1];
    scompressed += ".compressed";
    FILE *compressedFilePtr = fopen(&scompressed[0], "wb");

    int bitCounter = 0;
    unsigned char bufferByte;
    // Writing the first piece of header information: the count of unique bytes.
    // This count is essential for reconstructing the Huffman tree during the decompression process.
    fwrite(&letter_count, 1, 1, compressedFilePtr);
    // Update the totalBitCount to include the 8 bits (1 byte) just written for the unique byte count.
    totalBitCount += 8;

    //----------------------------------------

    // Initializing a pointer for iterating through the transformation strings.
    char *transformationStringPtr;
    // Variables for storing the length of the transformation string and the current character being processed.
    unsigned char transformationLength, currentCharacter;
    // Array to store transformation strings for each unique character to optimize compression.
    string transformationStrings[256];

    // Iterate through each node in the Huffman tree to write transformation codes to the compressed file.
    for (TreeNode *node = nodesForHuffmanTree; node < nodesForHuffmanTree + letter_count; node++)
    {
        // Store the transformation string for the current character in the array.
        transformationStrings[node->character] = node->bit;
        transformationLength = node->bit.length();
        currentCharacter = node->character;

        // Write the current character and its transformation string length to the compressed file.
        write_from_uChar(currentCharacter, bufferByte, bitCounter, compressedFilePtr);
        write_from_uChar(transformationLength, bufferByte, bitCounter, compressedFilePtr);
        // Updating the total bit count to include the bits for the character and its length.
        totalBitCount += transformationLength + 16;

        // Write the transformation string bit by bit to the compressed file.
        transformationStringPtr = &node->bit[0];
        while (*transformationStringPtr)
        {
            if (bitCounter == 8)
            {
                fwrite(&bufferByte, 1, 1, compressedFilePtr);
                bitCounter = 0;
            }
            if (*transformationStringPtr == '1')
            {
                bufferByte <<= 1;
                bufferByte |= 1;
            }
            else
            {
                bufferByte <<= 1;
            }
            bitCounter++;
            transformationStringPtr++;
        }

        // Adjust the total bit count based on the occurrences of the current character.
        totalBitCount += transformationLength * (node->occurances);
    }

    // Adjust the total bit count to be a multiple of 8, ensuring it represents the total number of bytes used.
    if (totalBitCount % 8)
    {
        totalBitCount = (totalBitCount / 8 + 1) * 8;
    }
    // This loop processes the Huffman tree nodes, writing their associated transformation scripts to the compressed file.
    //----------------------------------------

    std::cout << "The size of the sum of ORIGINAL files is: " << totalSize << " bytes" << endl;
    std::cout << "The size of the COMPRESSED file will be: " << totalBitCount / 8 << " bytes" << endl;

    float compressionRatio = 100.0f * static_cast<float>(totalBitCount) / 8.0f / static_cast<float>(totalSize);
    std::cout << "Compressed file's size will be approximately [" << compressionRatio << "%] of the original files." << endl;

    // Warning if the compressed file is unexpectedly larger than the original sum.
    if (totalBitCount / 8 > totalSize)
    {
        std::cout << "\nWARNING: The compressed file's size is larger than the sum of the originals.\n\n";
    }

    // Setting the progress bar's maximum value to the total occurrences of all
    // characters, represented by the root node of the Huffman tree. This reflects the
    // total number of characters processed during compression, providing a measure for
    // tracking compression progress.
    PROGRESS.MAX = (nodesForHuffmanTree + letter_count * 2 - 2)->occurances;

    // Writing the total count of files to be compressed to the header of the
    // compressed file.
    write_file_count(1, bufferByte, bitCounter, compressedFilePtr);
    //---------------------------------------

    originalFilePtr = fopen(argv[1], "rb");
    // Moving to the end of the file to determine its size.
    fseek(originalFilePtr, 0, SEEK_END);
    size = ftell(originalFilePtr);
    // Rewinding to the start of the file to read from the beginning.
    rewind(originalFilePtr);

    // Handling bit alignment before writing file information.
    if (bitCounter == 8)
    {
        fwrite(&bufferByte, 1, 1, compressedFilePtr);
        bitCounter = 0;
    }
    bufferByte <<= 1;
    bufferByte |= 1;
    bitCounter++;

    // Writing the size of the file, its name, and its content in the compressed format.
    write_file_size(size, bufferByte, bitCounter, compressedFilePtr);
    write_file_name(argv[1], transformationStrings, bufferByte, bitCounter, compressedFilePtr);
    write_the_file_content(originalFilePtr, size, transformationStrings, bufferByte, bitCounter, compressedFilePtr);
    fclose(originalFilePtr);

    // Ensuring the last byte is written to the compressed file by aligning the bit counter.
    if (bitCounter > 0)
    {
        bufferByte <<= (8 - bitCounter);
        fwrite(&bufferByte, 1, 1, compressedFilePtr);
    }

    fclose(compressedFilePtr);
    std::cout << endl
              << "Created compressed file: " << scompressed << endl;
    std::cout << "Compression is complete" << endl;
}

// below function is used for writing the uChar to compressed file
// It does not write it directly as one byte instead it mixes uChar and current byte, writes 8 bits of it
// and puts the rest to curent byte for later use
void write_from_uChar(unsigned char uChar, unsigned char &current_byte, int current_bit_count, FILE *fp_write)
{
    current_byte <<= 8 - current_bit_count;
    current_byte |= (uChar >> current_bit_count);
    fwrite(&current_byte, 1, 1, fp_write);
    current_byte = uChar;
}

// below function is writing number of files we re going to translate inside current folder to compressed file's 2 bytes
// It is done like this to make sure that it can work on little, big or middle-endian systems
void write_file_count(int file_count, unsigned char &current_byte, int current_bit_count, FILE *compressed_fp)
{
    unsigned char temp = file_count % 256;
    write_from_uChar(temp, current_byte, current_bit_count, compressed_fp);
    temp = file_count / 256;
    write_from_uChar(temp, current_byte, current_bit_count, compressed_fp);
}

// This function is writing byte count of current input file to compressed file using 8 bytes
// It is done like this to make sure that it can work on little, big or middle-endian systems
void write_file_size(long int size, unsigned char &current_byte, int current_bit_count, FILE *compressed_fp)
{
    PROGRESS.next(size); // updating progress bar
    for (int i = 0; i < 8; i++)
    {
        write_from_uChar(size % 256, current_byte, current_bit_count, compressed_fp);
        size /= 256;
    }
}

// This function writes bytes that are translated from current input file's name to the compressed file.
void write_file_name(char *file_name, string *str_arr, unsigned char &current_byte, int &current_bit_count, FILE *compressed_fp)
{
    write_from_uChar(strlen(file_name), current_byte, current_bit_count, compressed_fp);
    char *str_pointer;
    for (char *c = file_name; *c; c++)
    {
        str_pointer = &str_arr[(unsigned char)(*c)][0];
        while (*str_pointer)
        {
            if (current_bit_count == 8)
            {
                fwrite(&current_byte, 1, 1, compressed_fp);
                current_bit_count = 0;
            }
            switch (*str_pointer)
            {
            case '1':
                current_byte <<= 1;
                current_byte |= 1;
                current_bit_count++;
                break;
            case '0':
                current_byte <<= 1;
                current_bit_count++;
                break;
            default:
                cout << "An error has occurred" << endl
                     << "Process has been aborted";
                exit(2);
            }
            str_pointer++;
        }
    }
}

// Below function translates and writes bytes from current input file to the compressed file.
void write_the_file_content(FILE *original_fp, long int size, string *str_arr, unsigned char &current_byte, int &current_bit_count, FILE *compressed_fp)
{
    unsigned char *x_p, x;
    x_p = &x;
    char *str_pointer;
    fread(x_p, 1, 1, original_fp);
    for (long int i = 0; i < size; i++)
    {
        str_pointer = &str_arr[x][0];
        while (*str_pointer)
        {
            if (current_bit_count == 8)
            {
                fwrite(&current_byte, 1, 1, compressed_fp);
                current_bit_count = 0;
            }
            switch (*str_pointer)
            {
            case '1':
                current_byte <<= 1;
                current_byte |= 1;
                current_bit_count++;
                break;
            case '0':
                current_byte <<= 1;
                current_bit_count++;
                break;
            default:
                cout << "An error has occurred" << endl
                     << "Process has been aborted";
                exit(2);
            }
            str_pointer++;
        }
        fread(x_p, 1, 1, original_fp);
    }
}

int this_is_not_a_folder(char *path)
{
    DIR *temp = opendir(path);
    if (temp)
    {
        closedir(temp);
        return 0;
    }
    return 1;
}

long int size_of_the_file(char *path)
{
    long int size;
    FILE *fp = fopen(path, "rb");
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fclose(fp);
    return size;
}

// This function counts usage frequency of bytes inside a folder
// only give folder path as input
void count_in_folder(string path, long int *number, long int &total_size, long int &total_bits)
{
    FILE *original_fp;
    path += '/';
    DIR *dir = opendir(&path[0]), *next_dir;
    string next_path;
    total_size += 4096;
    total_bits += 16; // for file_count
    struct dirent *current;
    while ((current = readdir(dir)))
    {
        if (current->d_name[0] == '.')
        {
            if (current->d_name[1] == 0)
                continue;
            if (current->d_name[1] == '.' && current->d_name[2] == 0)
                continue;
        }
        total_bits += 9;

        for (char *c = current->d_name; *c; c++)
        { // counting usage frequency of bytes on the file name (or folder name)
            number[(unsigned char)(*c)]++;
        }

        next_path = path + current->d_name;

        if ((next_dir = opendir(&next_path[0])))
        {
            closedir(next_dir);
            count_in_folder(next_path, number, total_size, total_bits);
        }
        else
        {
            long int size;
            unsigned char *x_p, x;
            x_p = &x;
            total_size += size = size_of_the_file(&next_path[0]);
            total_bits += 64;

            //--------------------2------------------------
            original_fp = fopen(&next_path[0], "rb");

            fread(x_p, 1, 1, original_fp);
            for (long int j = 0; j < size; j++)
            { // counting usage frequency of bytes inside the file
                number[x]++;
                fread(x_p, 1, 1, original_fp);
            }
            fclose(original_fp);
        }
    }
    closedir(dir);
}

void write_the_folder(string path, string *str_arr, unsigned char &current_byte, int &current_bit_count, FILE *compressed_fp)
{
    FILE *original_fp;
    path += '/';
    DIR *dir = opendir(&path[0]), *next_dir;
    string next_path;
    struct dirent *current;
    int file_count = 0;
    long int size;
    while ((current = readdir(dir)))
    {
        if (current->d_name[0] == '.')
        {
            if (current->d_name[1] == 0)
                continue;
            if (current->d_name[1] == '.' && current->d_name[2] == 0)
                continue;
        }
        file_count++;
    }
    rewinddir(dir);
    write_file_count(file_count, current_byte, current_bit_count, compressed_fp);

    while ((current = readdir(dir)))
    { // if current is a file
        if (current->d_name[0] == '.')
        {
            if (current->d_name[1] == 0)
                continue;
            if (current->d_name[1] == '.' && current->d_name[2] == 0)
                continue;
        }

        next_path = path + current->d_name;
        if (this_is_not_a_folder(&next_path[0]))
        {

            original_fp = fopen(&next_path[0], "rb");
            fseek(original_fp, 0, SEEK_END);
            size = ftell(original_fp);
            rewind(original_fp);

            //-------------writes fifth--------------
            if (current_bit_count == 8)
            {
                fwrite(&current_byte, 1, 1, compressed_fp);
                current_bit_count = 0;
            }
            current_byte <<= 1;
            current_byte |= 1;
            current_bit_count++;
            //---------------------------------------

            write_file_size(size, current_byte, current_bit_count, compressed_fp);                              // writes sixth
            write_file_name(current->d_name, str_arr, current_byte, current_bit_count, compressed_fp);          // writes seventh
            write_the_file_content(original_fp, size, str_arr, current_byte, current_bit_count, compressed_fp); // writes eighth
            fclose(original_fp);
        }
        else
        { // if current is a folder

            //-------------writes fifth--------------
            if (current_bit_count == 8)
            {
                fwrite(&current_byte, 1, 1, compressed_fp);
                current_bit_count = 0;
            }
            current_byte <<= 1;
            current_bit_count++;
            //---------------------------------------

            write_file_name(current->d_name, str_arr, current_byte, current_bit_count, compressed_fp); // writes seventh

            write_the_folder(next_path, str_arr, current_byte, current_bit_count, compressed_fp);
        }
    }
    closedir(dir);
}