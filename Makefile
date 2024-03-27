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
	@rm -rf DECOMPRESSED_FILE*
	@echo "Testing archive"
	@./archive romeo.txt
	@echo "Testing extract"
	@./extract romeo.txt.compressed
	@echo "Comparing files"
	@diff romeo.txt DECOMPRESSED_FILE
	@echo "Delete compressed file"
	@rm romeo.txt.compressed
	@rm DECOMPRESSED_FILE*

.PHONY : all clean
