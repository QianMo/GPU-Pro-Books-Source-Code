Figure 1.3: Basic optimizations.
The "Unoptimized" curve was made with the implementation in 01_unoptimized.
The "Cache Friendly" curve was made with the implementation in 02_cache_friendly.
The "Cache Friendly and 16x16 Tiles" curve was made with the implementation in 03_cache_friendly_16x16_baseline.

Figure 1.5: Tiled culling optimizations.
The "Baseline" curve was made with the implementation in 03_cache_friendly_16x16_baseline.
The "Half Z" curve was made with the implementation in 04_halfz.
The "Modified Half Z" curve was made with the implementation in 05_modified_halfz.
The "Modified Half Z, AABB" curve was made with the implementation in 06_modified_halfz_aabb.
The "Modified Half Z, AABB, Parallel Reduction" curve was made with the implementation in 07_modified_halfz_aabb_parallel_reduction, 
which matches the final implementation in the ComputeBasedTiledCulling folder.

All curves in Figure 1.3 and Figure 1.5 were made with the following settings:
AMD Radeon(TM) R7 260X GPU
1920x1080 Fullscreen
Forward+
Triangle Density: High
Active Grid Objects: 280

With these settings, pressing the F10 key will cause a file named "<gpu_name> fp_1920x1080_ms1_grd280high.csv" to be written 
to the current working directory (typically the ComputeBasedTiledCulling directory).

Figure 1.3 used just the values from the "Tiled Culling" column, which is the tiled culling compute shader execution time.
Figure 1.5 used the sum of the "Tiled Culling" column and the "Forward Rendering" column.

Note that, for the "Modified Half Z, AABB, Parallel Reduction" curve in Figure 1.5, the "Tiled Culling" column also includes 
the execution time for the separate parallel reduction compute shader.
