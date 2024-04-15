#include <iostream>
#include <cstdio>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <dirent.h>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <sys/time.h>

using namespace std;

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

struct TreeNode
{ // this structure will be used to create the translation tree
    TreeNode *left, *right;
    long int occurrences;
    unsigned short character;
    string bit;
};

struct bufferedOutStream{
    unsigned char bufferByte = 0;
    int bitCounter = 0;

    std::vector<unsigned char> content;

    // bufferedOutStream(size_t approxSize) : content(approxSize) {
    //     content.reserve(approxSize);
    // }
    // bufferedOutStream() = default;

    void pushUChar(unsigned char byteToPush){
        bufferByte <<= 8 - bitCounter;
        bufferByte |= (byteToPush >> bitCounter);
        content.push_back(bufferByte);
        bufferByte = byteToPush;
    }

    template<typename T>
    void pushNByte(const T* dataPointer, int n){
        // Push n bytes of any data as char to contents
        // TODO: Not safe.
        const unsigned char* bytePointer = reinterpret_cast<const unsigned char*>(dataPointer);
        for (int i = 0; i < n; i++) {
            pushUChar(bytePointer[i]);
        }
    }

    void pushUShort(unsigned short shortToPush){
        unsigned char firstByte = (shortToPush >> 8) & 0xFF; // High byte
        unsigned char secondByte = shortToPush & 0xFF;       // Low byte
        pushUChar(firstByte);
        pushUChar(secondByte);
    }

    void pushFileSize(long int fileSize)
    {
        std::cout << "Pushing Filesize to Outstream. fileSize: " << fileSize <<std::endl;

        for (int i = 0; i < 8; i++)
        {
            pushUChar(fileSize % 256);
            fileSize /= 256;
        }
    }   

    void pushIfFullBuffer(){
        if (bitCounter == 8)
        {
            content.push_back(bufferByte);
            bitCounter = 0;
        }
    }

    void align(){
        if (bitCounter > 0) {
            // Align the last byte if it's not full.
            bufferByte <<= (8 - bitCounter);
            content.push_back(bufferByte);
        }
    }
    void flushToFile(FILE* filePtr) {
        align();
        // Write the accumulated content to the file.
        if (!content.empty()) {
            fwrite(&content[0], 1, content.size(), filePtr);
            std::cout << "Flushing content. Size: " << content.size() << "bytes."<<std::endl;
        }
        // fwrite(&content[0], 1, 3533, filePtr);
        // Reset the buffer and bit counter after flushing.
        content.clear();
        bitCounter = 0;
        bufferByte = 0;
    }
    void append(const bufferedOutStream& source) {

    if (bitCounter > 0) {
        
        for (int i = 0; i < source.bitCounter; ++i) {
            
            unsigned char bit = (source.bufferByte >> (7 - i)) & 1;
            bufferByte = (bufferByte << 1) | bit;
            bitCounter++;


            if (bitCounter == 8) {
                content.push_back(bufferByte);
                bufferByte = 0;
                bitCounter = 0;
            }
        }
        
        for (auto byte : source.content) {
            pushUChar(byte);
        }
    } else {
        
        content.insert(content.end(), source.content.begin(), source.content.end());


        if (source.bitCounter > 0) {
            bufferByte = source.bufferByte;
            bitCounter = source.bitCounter;
        }
    }
}
};

long int sizeOfTheFile(char *);

void cpuGenHuffmanTree(TreeNode *, int, long int *);
void cpuEncode(bufferedOutStream&, ::vector<unsigned char>, int, string *);
void ompEncode(bufferedOutStream &, ::vector<unsigned char> , int , string *);

bool TreeNodeCompare(TreeNode a, TreeNode b)
{
    return a.occurrences < b.occurrences;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Must provide a single file name." << endl;
        return 0;
    }

    ifstream originalFile(argv[1], ios::binary);
    if (!originalFile.is_open())
    {
        std::cout << argv[1] << " file does not exist" << endl
                  << "Process has been terminated" << endl;
        return 0;
    }

    originalFile.seekg(0, ios::end);
    long int originalFileSize = originalFile.tellg();
    originalFile.seekg(0, ios::beg);
    std::cout << "The size of the sum of ORIGINAL files is: " << originalFileSize << " bytes" << endl;

    long int freqCount[65536] = {0};
    int uniqueSymbolCount = 0;
    // Histograming the frequency of bytes.
    unsigned short readBuf;
    bool isOdd = originalFileSize % 2 == 1;
    unsigned char lastByte = 0;

    std::vector<unsigned char> fileData(originalFileSize);
    originalFile.read(reinterpret_cast<char *>(&fileData[0]), originalFileSize);
    originalFile.close();


    double totalBegin = getTimeStamp();


    double histoBegin = getTimeStamp();

    #pragma omp parallel for
    for (int i = 0; i < originalFileSize / 2; i++)
    {
        readBuf = (fileData[i * 2 + 1] << 8) | fileData[i * 2];
        freqCount[readBuf]++;
    }

    if (isOdd)
    {
        lastByte = fileData[originalFileSize - 1];
    }

    double histoEnd = getTimeStamp();
    printf("Histo time: %.3f ms\n", (histoEnd-histoBegin) * 1000.0);
    
    for (int i = 0; i < 65536; i++)
    {
        if (freqCount[i])
        {
            uniqueSymbolCount++;
        }
    }

    std::cout << "Unique symbols count: " << uniqueSymbolCount << endl;

    // Step 1: Initialize the leaf nodes for Huffman tree construction.
    // Each leaf node represents a unique byte and its frequency in the input data.
    TreeNode *nodesForHuffmanTree = new TreeNode[uniqueSymbolCount * 2 - 1];

    double genHuffmanTreeBegin = getTimeStamp();

    cpuGenHuffmanTree(nodesForHuffmanTree, uniqueSymbolCount, freqCount);

    double genHuffmanTreeEnd = getTimeStamp();


    double treeConstructionElapsedTime = getTimeStamp() - genHuffmanTreeBegin;
    printf("construction time: %.3f ms, symbols/s: %.3f\n", treeConstructionElapsedTime * 1000.0,
           (float)(uniqueSymbolCount) / (treeConstructionElapsedTime * 1e-3));
    string scompressed = argv[1];
    scompressed += ".compressed";
    // FILE *compressedFilePtr = fopen(&scompressed[0], "wb");

    bufferedOutStream outStream;
    double cpuEncodeBegin = getTimeStamp();
    // size_t approxCompressedSize = static_cast<size_t>( originalFileSize * 0.3);

    // outStream.pushUShort(static_cast<unsigned short>(uniqueSymbolCount));
    outStream.pushNByte(&uniqueSymbolCount, 2);
    // Write last byte information.

    // fwrite(&isOdd, 1, 1, compressedFilePtr);
    outStream.pushNByte(&isOdd, 1);
    if (isOdd)
    {
        // Write the last byte.
        // fwrite(&lastByte, 1, 1, compressedFilePtr);
        outStream.pushUChar(lastByte);
    }

    outStream.bitCounter = 0;
    outStream.bufferByte = 0;
    // Array to store transformation strings for each unique character to optimize compression.
    string transformationStrings[65536];

    // Iterate through each node in the Huffman tree to write transformation codes to the compressed file.
    for (TreeNode *node = nodesForHuffmanTree; node < nodesForHuffmanTree + uniqueSymbolCount; node++)
    {
        // Store the transformation string for the current character in the array.
        transformationStrings[node->character] = node->bit;
        unsigned char transformationLength = node->bit.length();
        unsigned short currentCharacter = node->character;

        // Write the current character and its transformation string length to the compressed file.

        outStream.pushUShort(currentCharacter);
        outStream.pushUChar(transformationLength);

        // Write the transformation string bit by bit to the compressed file.
        char *transformationStringPtr = &node->bit[0];
        while (*transformationStringPtr)
        {
            outStream.bufferByte <<= 1;
            if (*transformationStringPtr == '1')
            {
                outStream.bufferByte |= 1;
            }
            outStream.bitCounter++;
            transformationStringPtr++;
            // writeIfFullBuffer(bufferByte, bitCounter, compressedFilePtr);
            outStream.pushIfFullBuffer();
        }
    }

    outStream.pushFileSize(originalFileSize);

    ompEncode(outStream, fileData, originalFileSize, transformationStrings);
    // cpuEncode(outStream, fileData, originalFileSize, transformationStrings);

    // Ensuring the last byte is written to the compressed file by aligning the bit counter.
    outStream.align();
    FILE *compressedFilePtr = fopen(&scompressed[0], "wb");
    outStream.flushToFile(compressedFilePtr);
    fclose(compressedFilePtr); 

    double cpuEncodeEnd = getTimeStamp();
    printf("Encoding time: %.3f ms\n", (cpuEncodeEnd-cpuEncodeBegin) * 1000.0);

    double totalEnd = getTimeStamp();
    printf("Total time: %.3f ms\n", (totalEnd-totalBegin) * 1000.0);
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

long int sizeOfTheFile(char *path)
{
    ifstream file(path, ifstream::ate | ifstream::binary);
    return file.tellg();
}

void writeIfFullBuffer(unsigned char &bufferByte, int &bitCounter, FILE *filePtr)
{
    if (bitCounter == 8)
    {
        fwrite(&bufferByte, 1, 1, filePtr);
        bitCounter = 0;
    }
}


void cpuGenHuffmanTree(TreeNode *nodesForHuffmanTree, int uniqueSymbolCount, long int *freqCount) {

    TreeNode *currentNode = nodesForHuffmanTree;
    for (long int *frequency = freqCount; frequency < freqCount + 65536; frequency++)
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
    }}


void cpuEncode(bufferedOutStream &outStream, ::vector<unsigned char> fileData, int originalFileSize, string *transformationStrings){
    
    // translate file content to outstream
    unsigned short currentWord = 0;
    char *strPointer;
    for (int i = 0; i < originalFileSize; i += 2)
    {   
        if (i + 1 < originalFileSize) 
        {
            currentWord = (fileData[i+1] << 8) | fileData[i];
        }
        strPointer = &transformationStrings[currentWord][0];
        while (*strPointer)
        {
            outStream.pushIfFullBuffer();

            outStream.bufferByte <<= 1;
            if (*strPointer == '1')
            {
                outStream.bufferByte |= 1;
            }
            outStream.bitCounter++;
            strPointer++;
        }
    }
}

void ompEncode(bufferedOutStream &outStream, ::vector<unsigned char> fileData, int originalFileSize, string *transformationStrings){
    
    // translate file content to outstream
    unsigned short currentWord = 0;
    char *strPointer;
    #pragma omp parallel for
    for (int i = 0; i < originalFileSize; i += 2)
    {
        unsigned short currentWord = 0;
        if (i + 1 < originalFileSize)
        {
            currentWord = (fileData[i+1] << 8) | fileData[i];
        }
        // Note: We only transform data here, we do not write to the output stream.
        string localTransformationString = transformationStrings[currentWord];
        // Storing the transformations in a thread-private or carefully managed shared structure could be the next step
    }

    #pragma omp parallel
    {
        bufferedOutStream localOutStream; // Assume we have a way to create a local buffer stream
        
        #pragma omp for nowait // Use nowait to avoid implicit synchronization at the end of the loop
        for (int i = 0; i < originalFileSize; i += 2)
        {
            unsigned short currentWord = 0;
            if (i + 1 < originalFileSize)
            {
                currentWord = (fileData[i+1] << 8) | fileData[i];
            }
            const char *strPointer = transformationStrings[currentWord].c_str();
            
            while (*strPointer)
            {
                localOutStream.pushIfFullBuffer();
                localOutStream.bufferByte <<= 1;
                if (*strPointer == '1')
                {
                    localOutStream.bufferByte |= 1;
                }
                localOutStream.bitCounter++;
                strPointer++;
            }
        }
        
        #pragma omp critical
        {
            outStream.append(localOutStream); // You would need a method to append a local stream to the main stream
        }
    }
}