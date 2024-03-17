all : archive extract

archive : Compressor.cu
	nvcc -o archive Compressor.cu

extract : Decompressor.cu
	nvcc -o extract Decompressor.cu

clean :
	@rm -f archive
	@rm -f extract

.PHONY : all clean
