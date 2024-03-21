#include <iostream>
#include <cstdio>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <dirent.h>
#include "progress_bar.hpp"

using namespace std;

void writeFromUChar(unsigned char, unsigned char &, int, FILE *);
long int sizeOfTheFile(char *);
void writeFileSize(long int, unsigned char &, int, FILE *);
void writeFileName(char *, string *, unsigned char &, int &, FILE *);
void writeFileContent(FILE *, long int, string *, unsigned char &, int &, FILE *);

progress PROGRESS;

struct TreeNode
{ // this structure will be used to create the translation tree
    TreeNode *left, *right;
    long int occurrences;
    unsigned char character;
    string bit;
};

bool TreeNodeCompare(TreeNode a, TreeNode b)
{
    return a.occurrences < b.occurrences;
}

int main(int argc, char *argv[])
{
    long int freqCount[256] = {0};
    int uniqueSymbolCount = 0;
    if (argc != 2)
    {
        std::cout << "Must provide a single file name." << endl;
        return 0;
    }

    FILE *originalFilePtr;
    originalFilePtr = fopen(argv[1], "rb");
    if (!originalFilePtr)
    {
        std::cout << argv[1] << " file does not exist" << endl
                  << "Process has been terminated" << endl;
        return 0;
    }
    fclose(originalFilePtr);

    // Histograming the frequency of bytes.
    unsigned char *readBufPtr, readBuf;
    readBufPtr = &readBuf;
    long int originalFileSize = 0;

    for (char *c = argv[1]; *c; c++)
    { // counting usage frequency of unique bytes on the file name
        freqCount[(unsigned char)(*c)]++;
    }

    originalFileSize = sizeOfTheFile(argv[1]);
    std::cout << "The size of the sum of ORIGINAL files is: " << originalFileSize << " bytes" << endl;

    // "rb" is for reading binary files
    originalFilePtr = fopen(argv[1], "rb");
    // reading the first byte of the file into readBuf.
    fread(readBufPtr, 1, 1, originalFilePtr);
    for (long int i = 0; i < originalFileSize; i++)
    { // counting usage frequency of unique bytes inside the file
        freqCount[readBuf]++;
        fread(readBufPtr, 1, 1, originalFilePtr);
    }
    fclose(originalFilePtr);

    // Traverse through all possible bytes and count the number of unique bytes.
    for (long int *i = freqCount; i < freqCount + 256; i++)
    {
        if (*i)
        {
            uniqueSymbolCount++;
        }
    }
    //---------------------------------------------

    //--------------------3------------------------
    // Step 1: Initialize the leaf nodes for Huffman tree construction.
    // Each leaf node represents a unique byte and its frequency in the input data.
    TreeNode nodesForHuffmanTree[uniqueSymbolCount * 2 - 1];
    TreeNode *currentNode = nodesForHuffmanTree;

    // Step 2: Fill the array with data for each unique byte.
    for (long int *frequency = freqCount; frequency < freqCount + 256; frequency++)
    {
        if (*frequency)
        {
            currentNode->right = NULL;
            currentNode->left = NULL;
            currentNode->occurrences = *frequency;
            currentNode->character = frequency - freqCount;
            currentNode++;
        }
    }

    // Step 3: Sort the leaf nodes based on frequency to prepare for tree construction.
    // In ascending order.
    sort(nodesForHuffmanTree, nodesForHuffmanTree + uniqueSymbolCount, TreeNodeCompare);

    // Step 4: Construct the Huffman tree by merging nodes with the lowest frequencies.
    TreeNode *smallestNode = nodesForHuffmanTree;
    TreeNode *secondSmallestNode = nodesForHuffmanTree + 1;
    TreeNode *newInternalNode = nodesForHuffmanTree + uniqueSymbolCount;
    TreeNode *nextInternalNode = nodesForHuffmanTree + uniqueSymbolCount;
    TreeNode *nextLeafNode = nodesForHuffmanTree + 2;
    for (int i = 0; i < uniqueSymbolCount - 1; i++)
    {
        // Create a new internal node that combines the two smallest nodes.
        newInternalNode->occurrences = smallestNode->occurrences + secondSmallestNode->occurrences;
        newInternalNode->left = smallestNode;
        newInternalNode->right = secondSmallestNode;
        // Assign bits for tree navigation: '1' for the path to smallestNode,
        // '0' for secondSmallestNode.
        smallestNode->bit = "1";
        secondSmallestNode->bit = "0";
        newInternalNode++;

        // Update smallestNode and secondSmallestNode for the next iteration.
        if (nextLeafNode >= nodesForHuffmanTree + uniqueSymbolCount)
        {
            // All leaf nodes have been processed; proceed with internal nodes.
            smallestNode = nextInternalNode;
            nextInternalNode++;
        }
        else
        {
            // Choose the next smallest node from the leaf or internal nodes.
            smallestNode = (nextLeafNode->occurrences < nextInternalNode->occurrences) ? nextLeafNode++ : nextInternalNode++;
        }

        // Repeat the process for secondSmallestNode.
        if (nextLeafNode >= nodesForHuffmanTree + uniqueSymbolCount)
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
            secondSmallestNode = (nextLeafNode->occurrences < nextInternalNode->occurrences) ? nextLeafNode++ : nextInternalNode++;
        }
    }

    // Step 5: Assign Huffman codes to each node.
    // Iterate from the last internal node to the root, building the Huffman codes in reverse.
    for (TreeNode *node = nodesForHuffmanTree + uniqueSymbolCount * 2 - 2; node > nodesForHuffmanTree - 1; node--)
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
    fwrite(&uniqueSymbolCount, 1, 1, compressedFilePtr);

    //----------------------------------------

    // Initializing a pointer for iterating through the transformation strings.
    char *transformationStringPtr;
    // Variables for storing the length of the transformation string and the current character being processed.
    unsigned char transformationLength, currentCharacter;
    // Array to store transformation strings for each unique character to optimize compression.
    string transformationStrings[256];

    // Iterate through each node in the Huffman tree to write transformation codes to the compressed file.
    for (TreeNode *node = nodesForHuffmanTree; node < nodesForHuffmanTree + uniqueSymbolCount; node++)
    {
        // Store the transformation string for the current character in the array.
        transformationStrings[node->character] = node->bit;
        transformationLength = node->bit.length();
        currentCharacter = node->character;

        // Write the current character and its transformation string length to the compressed file.
        writeFromUChar(currentCharacter, bufferByte, bitCounter, compressedFilePtr);
        writeFromUChar(transformationLength, bufferByte, bitCounter, compressedFilePtr);

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
    }

    // Setting the progress bar's maximum value to the total occurrences of all
    // characters, represented by the root node of the Huffman tree. This reflects the
    // total number of characters processed during compression, providing a measure for
    // tracking compression progress.
    PROGRESS.MAX = (nodesForHuffmanTree + uniqueSymbolCount * 2 - 2)->occurrences;

    originalFilePtr = fopen(argv[1], "rb");

    // Handling bit alignment before writing file information.
    if (bitCounter == 8)
    {
        fwrite(&bufferByte, 1, 1, compressedFilePtr);
        bitCounter = 0;
    }

    // Writing the size of the file, its name, and its content in the compressed format.
    writeFileSize(originalFileSize, bufferByte, bitCounter, compressedFilePtr);
    writeFileName(argv[1], transformationStrings, bufferByte, bitCounter, compressedFilePtr);
    writeFileContent(originalFilePtr, originalFileSize, transformationStrings, bufferByte, bitCounter, compressedFilePtr);
    fclose(originalFilePtr);

    // Ensuring the last byte is written to the compressed file by aligning the bit counter.
    if (bitCounter > 0)
    {
        bufferByte <<= (8 - bitCounter);
        fwrite(&bufferByte, 1, 1, compressedFilePtr);
    }

    fclose(compressedFilePtr);

    // Get the size of compressed file.
    long int compressedFileSize = sizeOfTheFile(&scompressed[0]);
    std::cout << "The size of the COMPRESSED file is: " << compressedFileSize << " bytes" << endl;

    // Calculate the compression ratio.
    float compressionRatio = 100.0f * static_cast<float>(compressedFileSize) / static_cast<float>(originalFileSize);
    std::cout << "Compressed file's size is [" << compressionRatio << "%] of the original files." << endl;

    // Warning if the compressed file is unexpectedly larger than the original sum.
    if (compressedFileSize > originalFileSize)
    {
        std::cout << "\nWARNING: The compressed file's size is larger than the sum of the originals.\n\n";
    }

    std::cout << endl
              << "Created compressed file: " << scompressed << endl;
    std::cout << "Compression is complete" << endl;
}

// below function is used for writing the uChar to compressed file
// It does not write it directly as one byte instead it mixes uChar and current byte, writes 8 bits of it
// and puts the rest to curent byte for later use
void writeFromUChar(unsigned char uChar, unsigned char &current_byte, int current_bit_count, FILE *fp_write)
{
    current_byte <<= 8 - current_bit_count;
    current_byte |= (uChar >> current_bit_count);
    fwrite(&current_byte, 1, 1, fp_write);
    current_byte = uChar;
}

// This function is writing byte count of current input file to compressed file using 8 bytes
// It is done like this to make sure that it can work on little, big or middle-endian systems
void writeFileSize(long int size, unsigned char &current_byte, int current_bit_count, FILE *compressed_fp)
{
    PROGRESS.next(size); // updating progress bar
    for (int i = 0; i < 8; i++)
    {
        writeFromUChar(size % 256, current_byte, current_bit_count, compressed_fp);
        size /= 256;
    }
}

// This function writes bytes that are translated from current input file's name to the compressed file.
void writeFileName(char *file_name, string *str_arr, unsigned char &current_byte, int &current_bit_count, FILE *compressed_fp)
{
    writeFromUChar(strlen(file_name), current_byte, current_bit_count, compressed_fp);
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
void writeFileContent(FILE *original_fp, long int size, string *str_arr, unsigned char &current_byte, int &current_bit_count, FILE *compressed_fp)
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

long int sizeOfTheFile(char *path)
{
    ifstream file(path, ifstream::ate | ifstream::binary);
    return file.tellg();
}