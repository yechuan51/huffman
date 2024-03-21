all : archive extract

archive : Compressor.cu
	nvcc -o archive Compressor.cu

extract : Decompressor.cu
	nvcc -o extract Decompressor.cu

clean :
	@rm -f archive
	@rm -f extract

test :
	@echo "Testing archive"
	@./archive README.md
	@echo "Testing extract"
	@./extract README.md.compressed
	@echo "Comparing files"
	@diff README.md README\(1\).md
	@echo "Delete compressed file"
	@rm README.md.compressed
	@rm README\(1\).md

.PHONY : all clean
