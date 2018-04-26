 Source code for the demo accompanying "GPU Pro" article 
 "Polygonal-Functional Hybrids for Computer Animation and Games"
 Contacts: deniskravtsov@gmail.com

 This file is derived from the NVIDIA CUDA SDK example 'marchingCubes'
 You will need  NVIDIA CUDA SDK release 2.1 and CUDPP 1.0a (be sure to
 have $(CUDA_INC_PATH) and $(NVSDKCUDA_ROOT) env variables set correctly)
 The easiest way would be to copy the directory with the source code to
 your "$(NVSDKCUDA_ROOT)/projects" directory, as some include paths are 
 relative (otherwise nvcc might have problems if your path has spaces in 
 it)

 Uses stbi-1.18 - public domain JPEG/PNG reader ( www.nothings.org/stb_image.c )