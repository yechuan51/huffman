CC=nvcc

all : archive extract

archive : Compressor.cu
	${CC} -O3 -std=c++11 -o $@ $< -arch sm_80 --ptxas-options=-v -I.

extract : Decompressor.cu
	${CC} -O3 -o extract Decompressor.cu

clean :
	@rm -f archive
	@rm -f extract

test :
	@echo "Build archive and extract"
	@make all
	@echo "Testing archive"
	@./archive README.md
	@echo "Testing extract"
	@./extract README.md.compressed
	@echo "Comparing files"
	@diff README.md DECOMPRESSED_FILE
	@echo "Delete compressed file"
	@rm README.md.compressed
	@rm DECOMPRESSED_FILE*

.PHONY : all clean
