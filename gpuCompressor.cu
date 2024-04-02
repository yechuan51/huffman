#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/time.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "gpuHuffmanConstruction.h"

#define BLOCK_SIZE 256

using namespace std;

void writeFromUShort(unsigned short, unsigned char &, int, FILE *);
void writeFromUChar(unsigned char, unsigned char &, int, FILE *);
long int sizeOfTheFile(char *);
void writeFileSize(long int, unsigned char &, int, FILE *);
void writeFileContent(FILE *, long int, string *, unsigned char &, int &,
                      FILE *);
void writeIfFullBuffer(unsigned char &, int &, FILE *);

struct TreeNode {  // this structure will be used to create the translation tree
    TreeNode *left, *right;
    unsigned int occurrences;
    unsigned short character;
    string bit;
};

bool TreeNodeCompare(TreeNode a, TreeNode b) {
    return a.occurrences < b.occurrences;
}

__global__ void calculateFrequency(const unsigned char *data, long int size,
                                   unsigned int *freqCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size / 2; i += stride) {
        unsigned short readBuf = (data[i * 2 + 1] << 8) | data[i * 2];
        atomicAdd(&freqCount[readBuf], 1);
    }
}

__global__ void populateCWLength(const unsigned char *data, long int size, const int* transformationLengths,
                            unsigned int* CWLengths){
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x;
     
    for (int i = index; i < size / 2; i += stride) {
        unsigned short symbol = (data[i * 2 + 1] << 8) | data[i * 2];
        CWLengths[i + 1] = transformationLengths[symbol]; // i + 1 to let CW_lengths[0] be bitCounter
    }
}

__global__ void findOffset(unsigned int *input, int n,
                        unsigned int *output) {
    __shared__ unsigned int temp[BLOCK_SIZE * 2];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    if (start + t < n)
        temp[t] = input[start + t];
    else
        temp[t] = 0;

    if (start + blockDim.x + t < n)
        temp[blockDim.x + t] = input[start + blockDim.x + t];
    else
        temp[blockDim.x + t] = 0;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (t + 1) * stride * 2 - 1;
        if (index < 2 * blockDim.x)
            temp[index] += temp[index - stride];
    }

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (t + 1) * stride * 2 - 1;
        if (index + stride < 2 * blockDim.x)
            temp[index + stride] += temp[index];
    }

    __syncthreads();
    if (start + t < n)
        output[start + t] = temp[t];
    if (start + blockDim.x + t < n)
        output[start + blockDim.x + t] = temp[blockDim.x + t];
}

//__global__ void addBlockSum(unsigned int *output, unsigned int *lastOutput, int blockSize, int n) {
__global__ void addBlockSum( unsigned int *input, int blockSize, int n,
                        unsigned int *output) {
    int blockSum = 0;
    int blockStart = blockIdx.x * blockSize * 2;
    if(blockStart < n){
        for(int i = blockStart - 1; i >= 0; i -= blockSize * 2){
            blockSum += input[i];
        }
        // First half of block
        int index = blockStart + threadIdx.x;
        if (index < n) {
            output[index] = input[index] + blockSum;
        }
        // Second half of block
        index += blockSize;
        if (index < n) {
            output[index] = input[index] + blockSum;
        }
    }

    __syncthreads();
}

__device__ int binarySearch(const unsigned int* offsets, int numOffsets, int target){
    int low = 0;
    int high = numOffsets - 1;
    int leftIndex = -1;

    while (low <= high){
        int mid = low + (high - low) / 2;
        
        if(offsets[mid] == target){
            return mid;
        }
        else if(offsets[mid] < target){
            //update leftIndex and search upper half
            leftIndex = mid;
            low = mid + 1;
        } else {
            //search the lower half
            high = mid - 1;
        }
    }
    return leftIndex;
}

__global__ void encodeFromCW(const unsigned char *data, long int originalFileSize, unsigned char bufferByte,
                            char* transformationStrings, const int* transformationLengths, const int *transformationStringsOffset, 
                            const unsigned int *CW_offsets, const unsigned int* CW_lengths, const long int compressedFileSize, 
                            uint8_t *outputs) {
    //extern __shared__ int sdata[];
    // int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x; // example index = 88, index * 8 = 704

    uint8_t out = 0;
    if(index * 8 < compressedFileSize){
        // These two index can be used 
        int offset_index_left = binarySearch(CW_offsets, originalFileSize/2, index*8); // examply CW_offsets[offset_index_start] = 697
        int offset_index_right = offset_index_left + 1; // CW_offsets[offset_index_end] = example 709

        //printf("%d, %d, %d, %d, %d\n", index*8, offset_index_left, offset_index_right, CW_offsets[offset_index_left], CW_offsets[offset_index_right]);
        if(offset_index_left >= 0){
            // the transformation string related to the character to the left of index in offsets array
            unsigned short symbol_left = (data[offset_index_left * 2 + 1] << 8) | data[offset_index_left * 2];
            int t_string_left_offset = transformationStringsOffset[symbol_left];
            int t_string_left_length = transformationLengths[symbol_left];
            if( offset_index_right < originalFileSize/2 ){
                // the transformation string related to the character to the right of index in offsets array
                unsigned short symbol_right = (data[offset_index_right * 2 + 1] << 8) | data[offset_index_right * 2];
                int t_string_right_offset = transformationStringsOffset[symbol_right];
                int t_string_right_length = transformationLengths[symbol_right];
                //char* t_string = transformationStrings[t_string_offset .. t_string_offset+t_string_length ];

                //printf("%d, %d, %d, %d, %d, %d\n", symbol_left, symbol_right, t_string_left_offset, t_string_left_length, t_string_right_offset, t_string_right_length);

                int n = CW_offsets[offset_index_right] - index*8;
                if (n <= 8) {
                    // Take the last n bits of the t_string_left.
                    int bitToTake = 8;
                    for(int i = n - 1; i >= 0; i--){
                        out = out << 1;
                        if(transformationStrings[t_string_left_offset + t_string_left_length - 1 - i] == '1'){
                            out = out | 1;
                        } 
                        bitToTake -= 1;

                    }
                    while(bitToTake > 0){
                        int rightSideLen = bitToTake <= CW_lengths[offset_index_right] ? bitToTake: CW_lengths[offset_index_right];
                        for(int i = 0; i < rightSideLen; i++){
                            out = out << 1;
                            if(transformationStrings[t_string_right_offset + i] == '1'){
                                out = out | 1;
                            }
                            bitToTake -= 1;
                        }
                        offset_index_right += 1; //Prepare for next while loop iteration IF needed
                    }            
                }
                else {
                    // Take 8 bits from t_string_left, starting from t_string_left_offset + left_shift
                    int left_shift = index - CW_offsets[offset_index_left];
                    for(int i = 0; i < 8; i++){
                        out = out << 1;
                        if(transformationStrings[t_string_left_offset + left_shift + i] == '1'){
                            out = out | 1;
                        }
                    }
                }
            } else {
                // The last CW
                int n = CW_offsets[offset_index_left] + CW_lengths[offset_index_left] - index;
                if (n <= 8) {
                    // Take the last n bits of the t_string_left.
                    for(int i = n - 1; i >= 0; i--){
                        out = out << 1;
                        if(transformationStrings[t_string_left_offset + t_string_left_length - 1 - i] == '1'){
                            out = out | 1;
                        } 
                    }
                } else {
                    // Take 8 bits from t_string_left, starting from t_string_left_offset + left_shift
                    int left_shift = index - CW_offsets[offset_index_left];
                    for(int i = 0; i < 8; i++){
                        out = out << 1;
                        if(transformationStrings[t_string_left_offset + left_shift + i] == '1'){
                            out = out | 1;
                        } 
                    }
                }

            }
            
        } else {
            // Only here if it's the first character (index = 0) and CW_offests[0] > 0
            unsigned short symbol_right = (data[1] << 8) | data[0];
            int t_string_right_offset = transformationStringsOffset[symbol_right];
            int t_string_right_length = transformationLengths[symbol_right];

            out = out | bufferByte;
            for(int i = 0; i < 8 - CW_offsets[0]; i++){
                out = out << 1;
                if(transformationStrings[t_string_right_offset + i] == '1'){
                    out = out | 1;
                }
            }
            //printf("when index = 0, t_string_right_offset = %d, t_string_right_length = %d, out = %d\n", t_string_right_offset, t_string_right_length, out);
        }
        // printf("%d\n", out);
        outputs[index] = out;
    }



    // Each thread will handle one byte in output, and "extract" data array based on the offsets and CW_length
    // For example:
    // thread_bit_offset = index * 8; // this thread will handle the output bits from (index*8) to (index*8 + 7)
    // thread_bit_offset_end = index * 8 + 7; // this thread will handle the output bits from (index*8) to (index*8 + 7)
    // Find the offset_index that this thread_bit_offset belongs to in the offsets array(lookup/search).
    // Using these index, we know which word it's trying to encode from the data (offsets_index * 2 | offsets_index * 2 + 1)
    // Use transformationStrings to find the binary of the data, and based on previous search, put a total of 8 bits in the output[index]

    // optimization could include each thread handle multiple bytes (using a for loop), but must be a full byte in each loop iteration.


    // Each thread loads two characters from file (since codeword is upto 16 bits), convert to corresponding CW
    // Then store it to shared memory
    // if (index < size / 2) {
    //     unsigned short symbol = (data[index * 2 + 1] << 8) | data[index * 2];
    //     unsigned char* cw_string = transformationStrings[symbol]; // say 10110
    //     int cw_int = 0;
    //     for(int i = 0; cw_string[i] != '\0'; i++){
    //         cw_int = (cw_int << 1) | cw_string[i];
    //     }
    //     //sdata[tid] = (index < size / 2) ? cw_int : "";
    // }
    //__syncthreads();

    // Do reduction in shared memory
    // for (int s = 1; s < blockDim.x; s *= 2) {
    //     if (tid % (2*s) == 0) {
    //         // for(int i = sdata[tid + s]; i > 0; i >> 1){
    //         //     //sdata[tid] = (sdata[tid] << 1) | (i | 1);
    //         // }
    //         //sdata[tid] 
    //     }
    // }
    
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Must provide a single file name." << endl;
        return 0;
    }

    static constexpr int kMaxSymbolSize = 65536;
    ifstream originalFile(argv[1], ios::binary);
    if (!originalFile.is_open()) {
        std::cout << argv[1] << " file does not exist" << endl
                  << "Process has been terminated" << endl;
        return 0;
    }

    originalFile.seekg(0, ios::end);
    long int originalFileSize = originalFile.tellg();
    originalFile.seekg(0, ios::beg);
    std::cout << "The size of the sum of ORIGINAL files is: "
              << originalFileSize << " bytes" << endl;

    // Histograming the frequency of bytes.
    bool isOdd = originalFileSize % 2 == 1;
    unsigned char lastByte = 0;

    std::vector<unsigned char> fileData(originalFileSize);
    originalFile.read(reinterpret_cast<char *>(&fileData[0]), originalFileSize);
    originalFile.close();

    if (isOdd) {
        lastByte = fileData[originalFileSize - 1];
    }

    unsigned char *d_fileData;
    unsigned int *d_freqCount;
    cudaMalloc(&d_fileData, originalFileSize * sizeof(unsigned char));
    cudaMalloc(&d_freqCount, kMaxSymbolSize * sizeof(unsigned int));
    cudaMemset(d_freqCount, 0, kMaxSymbolSize * sizeof(unsigned int));

    cudaMemcpy(d_fileData, fileData.data(),
               originalFileSize * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (originalFileSize / 2 + blockSize - 1) / blockSize;
    calculateFrequency<<<numBlocks, blockSize>>>(d_fileData, originalFileSize,
                                                 d_freqCount);

    std::vector<unsigned int> freqCount(kMaxSymbolSize);
    cudaMemcpy(freqCount.data(), d_freqCount,
               kMaxSymbolSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // TODO: d_fileData will need to be used again in encoding stage
    // cudaFree(d_fileData);

    unsigned int uniqueSymbolCount = 0;
    for (int i = 0; i < 65536; i++) {
        if (freqCount[i]) {
            uniqueSymbolCount++;
        }
    }
    std::cout << "Unique symbols count: " << uniqueSymbolCount << endl;

    // Free unused memory.
    fileData.clear();

    // TODO: Remove this
    // Step 1: Initialize the leaf nodes for Huffman tree construction.
    // Each leaf node represents a unique byte and its frequency in the input
    // data. TreeNode nodesForHuffmanTree[uniqueSymbolCount * 2 - 1];
    TreeNode *nodesForHuffmanTree = new TreeNode[uniqueSymbolCount * 2 - 1];
    TreeNode *currentNode = nodesForHuffmanTree;

    // Step 2: Fill the array with data for each unique byte.
    for (unsigned int *frequency = freqCount.data();
         frequency < freqCount.data() + kMaxSymbolSize; frequency++) {
        if (*frequency) {
            currentNode->right = NULL;
            currentNode->left = NULL;
            currentNode->occurrences = *frequency;
            currentNode->character = frequency - freqCount.data();
            currentNode++;
        }
    }

    // Step 3: Sort the leaf nodes based on frequency to prepare for tree
    // construction. In ascending order.
    sort(nodesForHuffmanTree, nodesForHuffmanTree + uniqueSymbolCount,
         TreeNodeCompare);

    std::sort(
        freqCount.begin(), freqCount.end(),
        [](unsigned int const &a, unsigned int const &b) { return a < b; });

    cudaMemcpy(d_freqCount, freqCount.data(),
               kMaxSymbolSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    static constexpr int kThreadsPerBlock = 256;

    // Step 1: Initialize workspace
    GpuHuffmanWorkspace workspace(uniqueSymbolCount, kThreadsPerBlock);

    // Step 2: Gpu Code Word construction
    auto codewords = gpuCodebookConstruction(
        d_freqCount + kMaxSymbolSize - uniqueSymbolCount, uniqueSymbolCount,
        std::move(workspace), 0);

    cuda_check(cudaFree(d_freqCount));

    string scompressed = argv[1];
    scompressed += ".compressed";
    FILE *compressedFilePtr = fopen(&scompressed[0], "wb");

    // Writing the first piece of header information: the count of unique bytes.
    // This count is essential for reconstructing the Huffman tree during the
    // decompression process.
    fwrite(&uniqueSymbolCount, 2, 1, compressedFilePtr);

    // Write last byte information.

    fwrite(&isOdd, 1, 1, compressedFilePtr);
    if (isOdd) {
        // Write the last byte.
        fwrite(&lastByte, 1, 1, compressedFilePtr);
    }

    int bitCounter = 0;
    unsigned char bufferByte = 0;
    // Array to store transformation strings for each unique character to
    // optimize compression.

    std::vector<std::string> transformationStrings(kMaxSymbolSize);
    // Iterate through each node in the Huffman tree to write transformation
    // codes to the compressed file.
    int nodeIndex = 0;
    for (auto const &CW : codewords) {
        // Store the transformation string for the current character in the
        // array.
        auto node = nodesForHuffmanTree[nodeIndex];
        nodeIndex++;
        auto character = node.character;
        transformationStrings[character] = CW;
        unsigned char transformationLength = CW.length();
        unsigned short currentCharacter = character;

        // Write the current character and its transformation string length
        // to the compressed file.
        writeFromUShort(currentCharacter, bufferByte, bitCounter,
                        compressedFilePtr);
        writeFromUChar(transformationLength, bufferByte, bitCounter,
                       compressedFilePtr);

        // Write the transformation string bit by bit to the compressed
        // file.
        char const *transformationStringPtr = &CW[0];
        while (*transformationStringPtr) {
            bufferByte <<= 1;
            if (*transformationStringPtr == '1') {
                bufferByte |= 1;
            }
            bitCounter++;
            transformationStringPtr++;
            writeIfFullBuffer(bufferByte, bitCounter, compressedFilePtr);
        }
    }

    // codewords should be used here to write compressed file

    std::vector<int> h_transformationLengths;
    std::vector<int> h_transformationStringOffsets;
    unsigned int* h_data_lengths = (unsigned int *) malloc((originalFileSize/2 + 1) * sizeof(int));
    unsigned int* h_last_CW_length = (unsigned int*) malloc(sizeof(int));   // For tracking the last CW length to calculate total size of compressed file
    unsigned int* h_offsets = (unsigned int *) malloc((originalFileSize/2 + 1) * sizeof(int));
    unsigned int* h_lastOffset = (unsigned int*) malloc(sizeof(int));   // For tracking the total offsets to calculate total size of compressed file
    uint8_t *h_encode_buffer;

    h_transformationLengths.reserve(transformationStrings.size()); // Reserve space to avoid unnecessary reallocations
    std::transform(transformationStrings.begin(), transformationStrings.end(), std::back_inserter(h_transformationLengths), [](const std::string& str) {
        return str.length();
    });

    long int totalChars = 0;
    for(const int len: h_transformationLengths){
        h_transformationStringOffsets.push_back(totalChars);
        totalChars += len; 
    }
    //std::cout << std::endl;
    std::cout << "num chars = " << totalChars << std::endl;
    int* d_transformationLengths;
    unsigned int* d_data_lengths;
    unsigned int* d_offsets_t; // Transitional array, which is used to calculate final d_offsets
    unsigned int* d_offsets; 
    char* d_transformationStringsPool;
    int* d_transformationStringOffsets;
    uint8_t *d_encode_buffer;
    //std::cout << "originalFileSize = " << originalFileSize << std::endl;
    cudaMalloc((void**) &d_transformationLengths, transformationStrings.size() * sizeof(int));
    cudaMalloc((void**) &d_data_lengths, (originalFileSize/2 + 1) * sizeof(int));
    cudaMalloc((void**) &d_offsets_t, (originalFileSize/2 + 1) * sizeof(int));
    cudaMalloc((void**) &d_offsets, (originalFileSize/2 + 1) * sizeof(int));
    cudaMalloc((void**) &d_transformationStringsPool, totalChars * sizeof(unsigned char));
    cudaMalloc((void**) &d_transformationStringOffsets, transformationStrings.size() * sizeof(int));

    cudaMemcpy(d_transformationLengths, h_transformationLengths.data(),
               transformationStrings.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    
    const char* h_transformationStringPool = std::accumulate(transformationStrings.begin(), transformationStrings.end(), std::string("")).c_str(); // Convert vector of strings to one string to be sent to the kernel
    // Unknown bug: if h_transformationStringPool is defined at where all the other host variables, it will be empty by the time i t reaches cudaMemcpy
    std::cout << "h_transformationStringPool: " << h_transformationStringPool << std::endl;
    cudaMemcpy(d_transformationStringsPool, h_transformationStringPool,
               totalChars * sizeof(char),
               cudaMemcpyHostToDevice);


    cudaMemcpy(d_transformationStringOffsets, h_transformationStringOffsets.data(),
               transformationStrings.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    //std::cout << "Starting Findoffset Kernel" << std::endl;
    std::cout << "bitCounter before encoding: " << bitCounter << std::endl;
    //std::cout << "bufferByte before encoding: " << bufferByte << std::endl;
    cudaMemcpy(&d_data_lengths[0], &bitCounter, sizeof(int), cudaMemcpyHostToDevice); // Set the first value to be the bitCounter for offset purposes
    populateCWLength<<<numBlocks, blockSize>>>(d_fileData, originalFileSize, d_transformationLengths, d_data_lengths);
    //cudaDeviceSynchronize();
    findOffset<<<numBlocks, blockSize>>>(d_data_lengths, (originalFileSize/2 + 1), d_offsets_t);
    cudaDeviceSynchronize();
    //addBlockSum<<<numBlocks, blockSize>>>(d_offsets, d_offsets_t, blockSize, originalFileSize/2);
    addBlockSum<<<numBlocks, blockSize>>>(d_offsets_t, blockSize, (originalFileSize/2 + 1), d_offsets);
    cudaDeviceSynchronize();
   
    cudaMemcpy( h_lastOffset, d_offsets + originalFileSize/2, sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy( h_last_CW_length, d_data_lengths + originalFileSize/2, sizeof(int), cudaMemcpyDeviceToHost );
    // // h_lastOffset + h_last_CW_length contains the number of bits that needs to be allocated.
    long int compressedContentFileSize = h_lastOffset[0] + h_last_CW_length[0];

    if(compressedContentFileSize % 8 != 0){
        compressedContentFileSize += 8 - (compressedContentFileSize % 8);
    }

    cudaMalloc((void**) &d_encode_buffer, compressedContentFileSize);
    h_encode_buffer = (uint8_t*) malloc(compressedContentFileSize);

    std::cout << "h_lastOffset[0] Size: " << h_lastOffset[0] << std::endl;
    std::cout << "h_last_CW_length[0] Size: " << h_last_CW_length[0] << std::endl;
    std::cout << "Compressed File Size: " << compressedContentFileSize << std::endl;

    // TODO: Remove this normally, use for debugging
	cuda_check(cudaMemcpy( h_data_lengths, d_data_lengths, (originalFileSize/2 + 1) * sizeof(int), cudaMemcpyDeviceToHost ));
	cuda_check(cudaMemcpy( h_offsets, d_offsets, (originalFileSize/2 + 1) * sizeof(int), cudaMemcpyDeviceToHost ));

    // Tested to be correct
    for(long int i = 0; i < (originalFileSize/2 + 1); i++){
        std::cout << h_data_lengths[i] << " ";
    }
    std::cout << std::endl;

    int sum = bitCounter;
    for(long int i = 0; i < (originalFileSize/2 + 1); i++){
        // std::cout << h_offsets[i] << " ";
        if (sum != h_offsets[i]){
            std::cout << std::endl;
            std::cout << "h_offsets[i] does not match sum (h_offsets[i], sum, index): (" << h_offsets[i] << ", " << sum << ", " << i << ")" << std::endl;
            break;
        }
        unsigned short readBuf = (fileData.data()[i * 2 + 1] << 8) | fileData.data()[i * 2];
        sum += h_transformationLengths[readBuf];
    }
    std::cout << "h_offsets[i] are as expected" << std::endl;


    encodeFromCW<<<numBlocks, blockSize>>>(d_fileData, originalFileSize, bufferByte,
                                            d_transformationStringsPool, d_transformationLengths, d_transformationStringOffsets, 
                                            d_offsets, d_data_lengths, compressedContentFileSize, 
                                            d_encode_buffer);
    cudaDeviceSynchronize();
    

    //std::cout << "Finished Findoffset Kernel" << std::endl;

    cuda_check(cudaMemcpy( h_encode_buffer, d_encode_buffer, compressedContentFileSize, cudaMemcpyDeviceToHost));



    std::cout << "Compressed binary: " << std::endl;
    for(long int i = 0; i < compressedContentFileSize / 8; i++){
        uint8_t temp = h_encode_buffer[i];
        std::bitset<8> x(temp);

        std::cout << x << " ";
        // std::cout << temp << " ";
    }
    std::cout << "end" << std::endl;

    FILE *originalFilePtr = fopen(argv[1], "rb");
    // Writing the size of the file, its name, and its content in the
    // compressed format.
    writeFileSize(originalFileSize, bufferByte, bitCounter, compressedFilePtr);
    //writeFileContent(h_encode_buffer, bufferByte, bitCounter, compressedFilePtr);
    writeFileContent(originalFilePtr, originalFileSize,
                     transformationStrings.data(), bufferByte, bitCounter,
                     compressedFilePtr);
    fclose(originalFilePtr);

    // Ensuring the last byte is written to the compressed file by aligning
    // the bit counter.
    if (bitCounter > 0) {
        bufferByte <<= (8 - bitCounter);
        fwrite(&bufferByte, 1, 1, compressedFilePtr);
    }

    fclose(compressedFilePtr);

    // Get the size of compressed file.
    long int compressedFileSize = sizeOfTheFile(&scompressed[0]);
    std::cout << "The size of the COMPRESSED file is: " << compressedFileSize
              << " bytes" << endl;

    // Calculate the compression ratio.
    float compressionRatio = 100.0f * static_cast<float>(compressedFileSize) /
                             static_cast<float>(originalFileSize);
    std::cout << "Compressed file's size is [" << compressionRatio
              << "%] of the original files." << endl;

    // Warning if the compressed file is unexpectedly larger than the
    // original sum.
    if (compressedFileSize > originalFileSize) {
        std::cout << "\nWARNING: The compressed file's size is larger than the "
                     "sum of the originals.\n\n";
    }

    std::cout << endl << "Created compressed file: " << scompressed << endl;
    std::cout << "Compression is complete" << endl;
}

// below function is used for writing the uChar to compressed file
// It does not write it directly as one byte instead it mixes uChar and current
// byte, writes 8 bits of it and puts the rest to curent byte for later use
void writeFromUChar(unsigned char byteToWrite, unsigned char &bufferByte,
                    int bitCounter, FILE *filePtr) {
    // Going to write at least 1 byte, first shift the bufferByte to the left
    // to make room for the new byte.
    bufferByte <<= 8 - bitCounter;
    bufferByte |= (byteToWrite >> bitCounter);
    fwrite(&bufferByte, 1, 1, filePtr);
    bufferByte = byteToWrite;
}

void writeFromUShort(unsigned short shortToWrite, unsigned char &bufferByte,
                     int bitCounter, FILE *filePtr) {
    unsigned char firstByte = (shortToWrite >> 8) & 0xFF;  // High byte
    unsigned char secondByte = shortToWrite & 0xFF;        // Low byte

    writeFromUChar(firstByte, bufferByte, bitCounter, filePtr);
    writeFromUChar(secondByte, bufferByte, bitCounter, filePtr);
}

// This function is writing byte count of current input file to compressed file
// using 8 bytes It is done like this to make sure that it can work on little,
// big or middle-endian systems
void writeFileSize(long int fileSize, unsigned char &bufferByte, int bitCounter,
                   FILE *filePtr) {
    for (int i = 0; i < 8; i++) {
        writeFromUChar(fileSize % 256, bufferByte, bitCounter, filePtr);
        fileSize /= 256;
    }
}

// Below function translates and writes bytes from current input file to the
// compressed file.
void writeFileContent(FILE *originalFilePtr, long int originalFileSize,
                      string *transformationStrings, unsigned char &bufferByte,
                      int &bitCounter, FILE *compressedFilePtr) {
    unsigned short readBuf;
    unsigned char *readBufPtr;
    readBufPtr = (unsigned char *)&readBuf;
    // While loop reads the original file for current character (readBufPtr)
    while (fread(readBufPtr, 2, 1, originalFilePtr)) {
        char *strPointer = &transformationStrings[readBuf][0];
        while (*strPointer) {
            writeIfFullBuffer(bufferByte, bitCounter, compressedFilePtr);

            bufferByte <<= 1;
            if (*strPointer == '1') {
                bufferByte |= 1;
            }
            bitCounter++;
            strPointer++;
        }
    }
}

    //writeFileContent(h_encode_buffer, bufferByte, bitCounter, compressedFilePtr);
void writeFileContent(uint8_t* encode_buffer, unsigned char &bufferByte, int&bitCounter, FILE *compressedFilePtr) {
    
}

long int sizeOfTheFile(char *path) {
    ifstream file(path, ifstream::ate | ifstream::binary);
    return file.tellg();
}

void writeIfFullBuffer(unsigned char &bufferByte, int &bitCounter,
                       FILE *filePtr) {
    if (bitCounter == 8) {
        fwrite(&bufferByte, 1, 1, filePtr);
        bitCounter = 0;
    }
}
