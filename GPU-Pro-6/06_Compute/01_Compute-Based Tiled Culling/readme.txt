This is the companion code to the "Compute-Based Tiled Culling" chapter in GPU Pro 6.

The final optimized implementation (Modified Half Z with AABBs using parallel reduction 
for the depth bounds) is in the ComputeBasedTiledCulling folder, along with Visual Studio 
solution and project files for compiling and running the code.

Other implementations in the chapter are in the OtherImplementations folder. You can diff this 
code to see the changes from one implementation to the next. To compile and run one of the other 
implementations, replace the corresponding files in the ComputeBasedTiledCulling folder.

For benchmarking, pressing the F10 key will auto-increment the number of point lights from 0 to 2048. 
The performance results are written to a csv file in the current working directory (typically the 
ComputeBasedTiledCulling directory).

Note that, for clarity and simplicity of the code, MSAA support was not implemented. For a similar 
sample that implements MSAA as well as shadows, virtual point lights (VPLs), and transparency, see 
the TiledLighting11 sample in the Radeon(TM) SDK:
http://developer.amd.com/tools-and-sdks/graphics-development/amd-radeon-sdk/

One thing not mentioned in the chapter is that it can be faster in a tiled deferred implementation 
to run the culling in a separate shader from the lighting. The lighting code in modern real-time 
rendering engines can use a lot of GPU registers, and this high general-purpose register (GPR) 
pressure can reduce the number of wavefronts available for the scheduler, which can reduce latency 
hiding. For more details on how register pressure can affect execution efficiency, see "CodeXL for 
game developers: How to analyze your HLSL for GCN":
http://developer.amd.com/community/blog/2014/05/16/codexl-game-developers-analyze-hlsl-gcn/

In these cases, rather than have the culling code limited to the lower number of wavefronts 
caused by the high GPR usage of the lighting code, it can be faster to run culling in its 
own shader so it can execute more efficiently. That is, the culling code does not use a high 
number of registers and can run with a high number of simultaneous wavefronts in flight for 
better latency hiding if it is run separately from lighting. However, this results in additional 
overhead from adding a new pass. Thus, it may or may not be faster, depending on the lighting code. 

For this sample, there is a "Separate Culling" check-box. Note that, because the lighting code 
in the sample is relatively simple, the separate pass for culling is slower than having culling 
and lighting in the same shader (at least on the AMD Radeon(TM) R7 260X GPU). However, for the 
Unreal Engine 4 results gathered with the "Infiltrator" demo, the separate pass was faster.
