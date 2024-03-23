#include <iostream>
#include <cstdio>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <dirent.h>

using namespace std;

void writeFromUChar(unsigned char, unsigned char &, int, FILE *);
long int sizeOfTheFile(char *);
void writeFileSize(long int, unsigned char &, int, FILE *);
void writeFileContent(FILE *, long int, string *, unsigned char &, int &, FILE *);

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

    long int originalFileSize = sizeOfTheFile(argv[1]);
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

    // Writing the first piece of header information: the count of unique bytes.
    // This count is essential for reconstructing the Huffman tree during the decompression process.
    fwrite(&uniqueSymbolCount, 1, 1, compressedFilePtr);

    int bitCounter = 0;
    unsigned char bufferByte;
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
            bufferByte <<= 1;
            if (*transformationStringPtr == '1')
            {
                bufferByte |= 1;
            }
            bitCounter++;
            transformationStringPtr++;
            if (bitCounter == 8)
            {
                fwrite(&bufferByte, 1, 1, compressedFilePtr);
                bitCounter = 0;
            }
        }
    }

    originalFilePtr = fopen(argv[1], "rb");

    // Writing the size of the file, its name, and its content in the compressed format.
    writeFileSize(originalFileSize, bufferByte, bitCounter, compressedFilePtr);
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
void writeFromUChar(unsigned char byteToWrite, unsigned char &bufferByte, int bitCounter, FILE *filePtr)
{
    // Going to write at least 1 byte, first shift the bufferByte to the left
    // to make room for the new byte.
    bufferByte <<= 8 - bitCounter;
    bufferByte |= (byteToWrite >> bitCounter);
    fwrite(&bufferByte, 1, 1, filePtr);
    bufferByte = byteToWrite;
}

// This function is writing byte count of current input file to compressed file using 8 bytes
// It is done like this to make sure that it can work on little, big or middle-endian systems
void writeFileSize(long int fileSize, unsigned char &bufferByte, int bitCounter, FILE *filePtr)
{
    for (int i = 0; i < 8; i++)
    {
        writeFromUChar(fileSize % 256, bufferByte, bitCounter, filePtr);
        fileSize /= 256;
    }
}

// Below function translates and writes bytes from current input file to the compressed file.
void writeFileContent(FILE *originalFilePtr, long int originalFileSize, string *transformationStrings, unsigned char &bufferByte, int &bitCounter, FILE *compressedFilePtr)
{
    unsigned char *bufPtr, buf;
    bufPtr = &buf;
    char *strPointer;
    fread(bufPtr, 1, 1, originalFilePtr);
    for (long int i = 0; i < originalFileSize; i++)
    {
        strPointer = &transformationStrings[buf][0];
        while (*strPointer)
        {
            if (bitCounter == 8)
            {
                fwrite(&bufferByte, 1, 1, compressedFilePtr);
                bitCounter = 0;
            }
            switch (*strPointer)
            {
            case '1':
                bufferByte <<= 1;
                bufferByte |= 1;
                bitCounter++;
                break;
            case '0':
                bufferByte <<= 1;
                bitCounter++;
                break;
            default:
                cout << "An error has occurred" << endl
                     << "Process has been aborted";
                exit(2);
            }
            strPointer++;
        }
        fread(bufPtr, 1, 1, originalFilePtr);
    }
}

long int sizeOfTheFile(char *path)
{
    ifstream file(path, ifstream::ate | ifstream::binary);
    return file.tellg();
}