#include <assert.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

static const bool pagelocked_mem = true;
static const bool verbose = false;
static const bool debug = true;
static const bool micro_benchmark = false;

#define cuda_check(_x)                                     \
    do {                                                   \
        cudaError_t _err = (_x);                           \
        if (_err != cudaSuccess) {                         \
            printf("Error: cuda error %s(%d) occurred.\n", \
                   cudaGetErrorString(_err), int(_err));   \
            abort();                                       \
        }                                                  \
    } while (0)

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

struct timer {
    timer(cudaStream_t stream = 0);
    ~timer();
    void start();
    float stop();

   private:
    cudaStream_t m_stream;
    cudaEvent_t m_start, m_stop;
};

timer::timer(cudaStream_t stream) : m_stream{stream} {
    cuda_check(cudaEventCreate(&m_start));
    cuda_check(cudaEventCreate(&m_stop));
}

timer::~timer() {
    cuda_check(cudaEventDestroy(m_start));
    cuda_check(cudaEventDestroy(m_stop));
}

void timer::start() { cudaEventRecord(m_start, m_stream); }

float timer::stop() {
    float time;
    cudaEventRecord(m_stop, m_stream);
    cudaEventSynchronize(m_stop);
    cudaEventElapsedTime(&time, m_start, m_stop);
    return time;
}

struct Node {
    int index;
    int left;
    int right;
    int parent;
};

struct Barrier {
    Barrier(int* count, int n, int* sense)
        : _count{count},
          _n{n},
          _sense{sense},
          _local_sense{1},
          _wait_thread{threadIdx.x == 0} {}

    __device__ __forceinline__ void fetch(int& state) {
        if (_wait_thread) {
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                         : "=r"(state)
                         : "l"(_sense));
        }
    }

    __device__ __forceinline__ void fence() {
        int s = _local_sense ^ 1;
        _local_sense = s;
        __shared__ int s_old;
        int state = 0;
        fetch(state);
        if (_wait_thread) {
            int old = atomicAdd(_count, 1);
            s_old = old;
        }
        __syncthreads();
        int l_old = s_old;
        if (l_old == _n - 1) {
            if (_wait_thread) {
                *_count = 0;
                *_sense = s;
            }
        } else {
            while (__syncthreads_and(state != s)) {
                fetch(state);
            }
            __syncthreads();
        }
    }
    int* _count;
    int _n;
    int* _sense;
    int _local_sense;
    bool _wait_thread;
};

template <typename F>
__host__ __device__ __forceinline__ int BinarySearch(F const* freq, int size,
                                                     F val) {
    int l = 0;
    int r = size - 1;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (freq[m] <= val) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

template <typename T>
__host__ __device__ __forceinline__ void Swap(T& left, T& right) {
    T tmp = left;
    left = right;
    right = tmp;
}

template <typename F>
__host__ __device__ __forceinline__ int2 KthElement(F const* leftFreq,
                                                    int leftSize,
                                                    F const* rightFreq,
                                                    int rightSize, int k) {
    int kth = k;
    int leftIndex = 0;
    int rightIndex = 0;
    leftSize = leftSize - 1;
    rightSize = rightSize - 1;

    for (;;) {
        if (leftSize == leftIndex) {
            return make_int2(leftIndex, rightIndex + kth);
        } else if (rightSize == rightIndex) {
            return make_int2(leftIndex + kth, rightIndex);
        }
        int mid1 = leftIndex + (leftSize - leftIndex) / 2;
        int mid2 = rightIndex + (rightSize - rightIndex) / 2;
        if (mid1 + mid2 < kth) {
            if (leftFreq[mid1] > rightFreq[mid2]) {
                rightIndex = mid2 + 1;
                kth = kth - mid2 - 1;
            } else {
                leftIndex = mid1 + 1;
                kth = kth - mid1 - 1;
            }
        } else {
            if (leftFreq[mid1] > rightFreq[mid2]) {
                leftSize = mid1;
            } else {
                rightSize = mid2;
            }
        }
    }
}

// template <typename F>
// __host__ __device__ __forceinline__ int2 KthElement(F const* leftFreq,
//                                                     int leftSize,
//                                                     F const* rightFreq,
//                                                     int rightSize, int k) {
//     int kth = k;
//     int thisIndex = 0;
//     int otherIndex = 0;
//     int thisSize = leftSize;
//     int otherSize = rightSize;
//     F const* thisFreq = leftFreq;
//     F const* otherFreq = rightFreq;
//
//     F pivot = 0;
//     int pivotIndex = -1;
//     for (;;) {
//         pivot = thisFreq[thisIndex];
//         pivotIndex = BinarySearch(otherFreq + otherIndex, otherSize, pivot);
//         if (pivotIndex < kth) {
//             if (pivotIndex == 0 && otherFreq[otherIndex] < pivot) {
//                 otherIndex = otherIndex + 1;
//                 thisIndex = thisIndex + kth - 1;
//                 break;
//             }
//             kth -= pivotIndex;
//             otherSize -= pivotIndex;
//             otherIndex += pivotIndex;
//         } else if (pivotIndex >= kth) {
//             otherIndex = otherIndex + kth;
//             break;
//         }
//         // printf("pivot: %d\n", pivot);
//         // printf("pivotIndex: %d\n", pivotIndex);
//         // printf("kth: %d\n", kth);
//         // printf("thisSize: %d\n", thisSize);
//         // printf("otherSize: %d\n", otherSize);
//         // printf("thisIndex: %d\n", thisIndex);
//         // printf("otherIndex: %d\n", otherIndex);
//         if (thisFreq[thisIndex] <= otherFreq[otherIndex]) {
//             Swap(thisFreq, otherFreq);
//             Swap(thisSize, otherSize);
//             Swap(thisIndex, otherIndex);
//         }
//     }
//     if (thisFreq == leftFreq) {
//         return make_int2(thisIndex, otherIndex);
//     } else {
//         return make_int2(otherIndex, thisIndex);
//     }
// }

template <typename F>
__device__ __forceinline__ void ParallelMerge(
    F const* leftFreq, int const* leftIndex, int leftSize, F const* rightFreq,
    int const* rightIndex, int rightSize, F* mergeFreq, int* mergeIndex,
    int participants, int mergePerThread, int2* partitionIndex,
    Barrier& barrier) {
    int bid = blockIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < participants) {
        int2 kth = make_int2(0, 0);
        if (tid >= 1) {
            kth = KthElement(leftFreq, leftSize, rightFreq, rightSize,
                             tid * mergePerThread);
        }
        partitionIndex[tid] = kth;
    }

    barrier.fence();
    if (tid < participants) {
        int2 start = partitionIndex[tid];
        int2 end = tid < participants - 1
                       ? partitionIndex[tid + 1]
                       : make_int2(leftSize - 1, rightSize - 1);
        int i = tid * mergePerThread;
        while (start.x < end.x or start.y < end.y) {
            if (start.x >= end.x) {
                mergeFreq[i] = rightFreq[start.y];
                mergeIndex[i] = rightIndex[start.y];
                start.y++;
                i++;
            } else if (start.y >= end.y) {
                mergeFreq[i] = leftFreq[start.x];
                mergeIndex[i] = leftIndex[start.x];
                start.x++;
                i++;
            } else if (leftFreq[start.x] < rightFreq[start.y]) {
                mergeFreq[i] = leftFreq[start.x];
                mergeIndex[i] = leftIndex[start.x];
                start.x++;
                i++;
            } else {
                mergeFreq[i] = leftFreq[start.y];
                mergeIndex[i] = leftIndex[start.y];
                start.y++;
                i++;
            }
        }
    }
}

template <typename F, int kThreadsPerBlock>
__global__ void GenerateCL(int* CL, F const* histogram, int size,
                           /* global variables */ F* nodeFreq, F* tempFreq,
                           int* nodeIndex, int* tempIndex, Node* nodes,
                           /* barrier */
                           int* count, int* sense,
                           /* parallel merge */ int2* partitionIndex,
                           /* CL legnth */ int* reduceCL, int* maxCL) {
    int bid = blockIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Barrier barrier{count, gridDim.x, sense};

    if (tid < size) {
        Node nd;
        nd.index = tid;
        nd.left = -1;
        nd.right = -1;
        nd.parent = -1;
        int4* pointer = reinterpret_cast<int4*>(nodes + tid);
        *pointer = reinterpret_cast<int4&>(nd);
        nodeFreq[tid] = histogram[tid];
        nodeIndex[tid] = tid;
    }
    int numCurrentNodes = size;
    barrier.fence();

    int sizeLeft = size;
    while (sizeLeft > 1) {
        F specFreq = nodeFreq[0] + nodeFreq[1];
        int pivot = BinarySearch(nodeFreq, sizeLeft, specFreq);

        pivot = pivot - pivot & 0x1;
        if (tid < sizeLeft - pivot) {
            tempFreq[tid] = nodeFreq[tid + pivot];
            tempIndex[tid] = nodeIndex[tid + pivot];
        }

        if (tid < (pivot >> 1)) {
            int left = nodeIndex[tid * 2];
            int right = nodeIndex[tid * 2 + 1];
            {
                Node node;
                *reinterpret_cast<int4*>(&node) =
                    *reinterpret_cast<int4*>(&nodes[left]);
                node.parent = numCurrentNodes + tid;
                *reinterpret_cast<int4*>(nodes + left) =
                    reinterpret_cast<int4&>(node);
            }
            {
                Node node;
                *reinterpret_cast<int4*>(&node) =
                    *reinterpret_cast<int4*>(&nodes[right]);
                node.parent = numCurrentNodes + tid;
                *reinterpret_cast<int4*>(nodes + right) =
                    reinterpret_cast<int4&>(node);
            }

            Node nd;
            nd.index = -1;
            nd.left = left;
            nd.right = right;
            nd.parent = -1;
            int4* pointer =
                reinterpret_cast<int4*>(nodes + numCurrentNodes + tid);
            *pointer = reinterpret_cast<int4&>(nd);
            tempFreq[sizeLeft - pivot + tid] =
                nodeFreq[tid * 2] + nodeFreq[tid * 2 + 1];
            tempIndex[sizeLeft - pivot + tid] = numCurrentNodes + tid;
        }
        numCurrentNodes += (pivot >> 1);

        barrier.fence();

        ParallelMerge(tempFreq, tempIndex, sizeLeft - pivot,
                      tempFreq + sizeLeft - pivot, tempIndex + sizeLeft - pivot,
                      (pivot >> 1), nodeFreq, nodeIndex);

        sizeLeft = sizeLeft - pivot + (pivot >> 1);
        barrier.fence();
    }

    int threadIndex = threadIdx.x;
    __shared__ int s_CL[kThreadsPerBlock];
    if (tid < size) {
        int4* pointer = reinterpret_cast<int4*>(nodes + tid);
        Node cur;
        *reinterpret_cast<int4*>(&cur) = *pointer;
        int parent = cur.parent;
        int length = 0;
        while (parent != -1) {
            length++;
            int4* pointer = reinterpret_cast<int4*>(nodes + parent);
            Node cur;
            *reinterpret_cast<int4*>(&cur) = *pointer;
            parent = cur.parent;
        }
        s_CL[threadIndex] = length;
        CL[tid] = length;
    }

    __syncthreads();
#define MAX(x, y) (x) < (y) ? (y) : (x)
    int val;
#pragma unroll
    for (int reduceBlock = (kThreadsPerBlock >> 1); reduceBlock >= 1;
         reduceBlock >>= 1) {
        if (reduceBlock >= 64 && threadIndex < reduceBlock) {
            s_CL[threadIndex] =
                MAX(s_CL[threadIndex], s_CL[threadIndex + reduceBlock]);
            __syncthreads();
        } else if (reduceBlock == 32) {
            s_CL[threadIndex] =
                MAX(s_CL[threadIndex], s_CL[threadIndex + reduceBlock]);
        } else {
            if (reduceBlock == 16) {
                val = s_CL[threadIndex];
            }
            val = MAX(val, __shfl_down_sync(0xffffffff, val, reduceBlock));
        }
    }
    if (threadIndex == 0) {
        reduceCL[bid] = val;
    }
    barrier.fence();
    // final round reduce
    if (bid == 0) {
        for (int i = threadIndex; i < gridDim.x; i += kThreadsPerBlock) {
            if (i == threadIndex) {
                s_CL[threadIndex] = reduceCL[i];
            } else {
                s_CL[threadIndex] = MAX(s_CL[threadIndex], reduceCL[i]);
            }
        }
        __syncthreads();
        int val;
#pragma unroll
        for (int reduceBlock = (kThreadsPerBlock >> 1); reduceBlock >= 1;
             reduceBlock >>= 1) {
            if (reduceBlock >= 64 && threadIndex < reduceBlock) {
                s_CL[threadIndex] =
                    MAX(s_CL[threadIndex], s_CL[threadIndex + reduceBlock]);
                __syncthreads();
            } else if (reduceBlock == 32) {
                s_CL[threadIndex] =
                    MAX(s_CL[threadIndex], s_CL[threadIndex + reduceBlock]);
            } else {
                if (reduceBlock == 16) {
                    val = s_CL[threadIndex];
                }
                val = MAX(val, __shfl_down_sync(0xffffffff, val, reduceBlock));
            }
        }
        if (threadIndex == 0) {
            *maxCL = val;
        }
    }
}

__global__ void GenerateCW(Node const* nodes, uint8_t* CW, int size,
                           int codeLengthPerWord) {
    int bid = blockIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int4 const* pointer = reinterpret_cast<int4 const*>(nodes + tid);
        Node cur;
        *reinterpret_cast<int4*>(&cur) = *pointer;
        uint8_t* cw = CW + tid * codeLengthPerWord;
        int child = tid;
        int parent = cur.parent;
        int length = 0;
        while (parent != -1) {
            int4 const* pointer = reinterpret_cast<int4 const*>(nodes + parent);
            Node cur;
            *reinterpret_cast<int4*>(&cur) = *pointer;
            if (child == cur.left) {
                cw[length] = 0;
            } else {
                cw[length] = 1;
            }
            child = parent;
            parent = cur.parent;
            length++;
        }
    }
}

void test_serial_merge() {
    int n = 100;
    int na = 10;
    int nb = 10;
    std::vector<int> a(na);
    std::vector<int> b(nb);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, n);
    for (auto& ele : a) {
        ele = dist(rng);
    }
    for (auto& ele : b) {
        ele = dist(rng);
    }
    for (int i = 0; i < a.size(); ++i) {
        printf("%d %c", a[i], i == a.size() - 1 ? '\n' : ' ');
    }
    for (int i = 0; i < b.size(); ++i) {
        printf("%d %c", b[i], i == b.size() - 1 ? '\n' : ' ');
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    for (int i = 0; i < a.size(); ++i) {
        printf("%d %c", a[i], i == a.size() - 1 ? '\n' : ' ');
    }
    for (int i = 0; i < b.size(); ++i) {
        printf("%d %c", b[i], i == b.size() - 1 ? '\n' : ' ');
    }

    int pivot = BinarySearch(a.data(), a.size(), 30);
    printf("val: 30, pos: %d\n", pivot);
    if (a[0] > b[0]) {
        int2 kth = KthElement(a.data(), a.size(), b.data(), b.size(), 4);
        printf("%d %d\n", kth.x, kth.y);
    } else {
        int2 kth = KthElement(b.data(), b.size(), a.data(), a.size(), 4);
        printf("%d %d\n", kth.y, kth.x);
    }

    if (a[0] > b[0]) {
        int2 kth = KthElement(a.data(), a.size(), b.data(), b.size(), 8);
        printf("%d %d\n", kth.x, kth.y);
    } else {
        int2 kth = KthElement(b.data(), b.size(), a.data(), a.size(), 8);
        printf("%d %d\n", kth.y, kth.x);
    }
}

__global__ void calculateFrequency(const unsigned char* data, long int size,
                                   unsigned int* freqCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size / 2; i += stride) {
        unsigned short readBuf = (data[i * 2 + 1] << 8) | data[i * 2];
        atomicAdd(&freqCount[readBuf], 1);
    }
}

class GpuCodewords {
   public:
    GpuCodewords(int codeLengthPerWord, int symbolSize)
        : _codeLengthPerWord(codeLengthPerWord), _symbolSize(symbolSize) {
        uint8_t* codewords;
        cuda_check(cudaMalloc(
            &codewords, sizeof(uint8_t) * codeLengthPerWord * symbolSize));
        auto CudaFree = [](uint8_t* pointer) { cuda_check(cudaFree(pointer)); };
        _codewords = {codewords, CudaFree};
    }

    std::vector<std::string> toCpu() {
        std::vector<std::string> transformationStrings;
        return transformationStrings;
    }

    uint8_t const* pointer() const { return _codewords.get(); }
    uint8_t* pointer() { return _codewords.get(); }

   private:
    int _codeLengthPerWord;
    int _symbolSize;
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> _codewords;
};

GpuCodewords gpuCodebookConstruction(unsigned int* frequencies,
                                     int symbolSize) {
    return GpuCodewords(8, symbolSize);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Must provide a single file name." << std::endl;
        return 0;
    }

    int symbolSize = 65536;

    std::ifstream originalFile(argv[1], std::ios::binary);
    if (!originalFile.is_open()) {
        std::cout << argv[1] << " file does not exist" << std::endl
                  << "Process has been terminated" << std::endl;
        return 0;
    }

    originalFile.seekg(0, std::ios::end);
    long int originalFileSize = originalFile.tellg();
    originalFile.seekg(0, std::ios::beg);
    std::cout << "The size of the sum of ORIGINAL files is: "
              << originalFileSize << " bytes" << std::endl;

    // Histograming the frequency of bytes.
    bool isOdd = originalFileSize % 2 == 1;
    unsigned char lastByte = 0;

    std::vector<unsigned char> fileData(originalFileSize);
    originalFile.read(reinterpret_cast<char*>(&fileData[0]), originalFileSize);
    originalFile.close();

    if (isOdd) {
        lastByte = fileData[originalFileSize - 1];
    }

    unsigned char* d_fileData;
    unsigned int* d_freqCount;
    cudaMalloc(&d_fileData, originalFileSize * sizeof(unsigned char));
    cudaMalloc(&d_freqCount, symbolSize * sizeof(unsigned int));
    cudaMemset(d_freqCount, 0, symbolSize * sizeof(unsigned int));

    cudaMemcpy(d_fileData, fileData.data(),
               originalFileSize * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (originalFileSize / 2 + blockSize - 1) / blockSize;
    calculateFrequency<<<numBlocks, blockSize>>>(d_fileData, originalFileSize,
                                                 d_freqCount);

    std::vector<unsigned> freqCount(symbolSize);
    cudaMemcpy(freqCount.data(), d_freqCount, symbolSize * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    std::sort(freqCount.begin(), freqCount.end());

    cudaFree(d_fileData);
    cudaFree(d_freqCount);
}
