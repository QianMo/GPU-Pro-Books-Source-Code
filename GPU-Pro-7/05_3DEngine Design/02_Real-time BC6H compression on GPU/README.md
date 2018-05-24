GPURealTimeBC6H
=======

Real-time BC6H compressor, which runs entirelly on GPU (implemented using DX11 and pixel shaders). Features two presets: 
* "Fast" - compresses a standard 256x256x6 envmap with a full mipmap chain in 0.07ms on AMD R9 270 (mid-range GPU).
* "Quality" - compresses a standard 256x256x6 envmap with a full mipmap chain in 3.913ms on AMD R9 270 (mid-range GPU). Compression quality is comparable to fast/normal presets of offline compressors.

Performance
===
Intel's BC6H compressor tested on Intel i7 860. GPURealTimeBC6H and DirectXTex (Compute Shader path) tested on AMD R9 270. Measured in MP/s.

         | GPU Real-Time BC6H "Fast" | GPU Real-Time BC6H "Quality"  | Intel "Very fast" | Intel "Fast" | Intel "Basic" | Intel "Slow" | Intel "Very slow" | DirectXTex 
:-------:|:-------------------------:|:-----------------------------:|:-----------------:|:------------:|:-------------:|:------------:|:-----------------:|:----------:
Atrium   | 8579.56                   | 157.91                        | 66.00             | 6.02         | 2.43          | 0.72         | 0.34              | 0.63       
Backyard | 8642.11                   | 159.91                        | 77.10             | 7.44         | 2.69          | 0.80         | 0.38              | 0.63       
Desk     | 8022.40                   | 143.55                        | 63.10             | 5.35         | 2.22          | 0.63         | 0.33              | 0.65       
Memorial | 7281.78                   | 154.26                        | 78.64             | 7.38         | 2.50          | 0.76         | 0.37              | 0.47       
Yucca    | 8809.17                   | 161.42                        | 74.98             | 7.13         | 2.83          | 0.83         | 0.39              | 0.73       
Average  | 8267.00                   | 155.41                        | 71.96             | 6.66         | 2.53          | 0.75         | 0.36              | 0.62       

Quality
===
Quality compared using RMSLE (lower is better).

         | GPU Real-Time BC6H "Fast" | GPU Real-Time BC6H "Quality"  | Intel "Very fast" | Intel "Fast" | Intel "Basic" | Intel "Slow" | Intel "Very slow" | DirectXTex 
:-------:|:-------------------------:|:-----------------------------:|:-----------------:|:------------:|:-------------:|:------------:|:-----------------:|:----------:
Atrium   | 0.0092                    | 0.0074                        | 0.008             | 0.0069       | 0.0067        | 0.0067       | 0.0067            | 0.0079     
Backyard | 0.0089                    | 0.0076                        | 0.0072            | 0.0067       | 0.0065        | 0.0065       | 0.0065            | 0.0075     
Desk     | 0.0552                    | 0.0333                        | 0.047             | 0.0307       | 0.0298        | 0.0294       | 0.0293            | 0.0413     
Memorial | 0.0194                    | 0.0135                        | 0.0192            | 0.0135       | 0.0133        | 0.0132       | 0.0131            | 0.0243      
Yucca    | 0.0189                    | 0.0122                        | 0.0145            | 0.0108       | 0.0105        | 0.0103       | 0.0103            | 0.0124     
Average  | 0.0223                    | 0.0148                        | 0.0192            | 0.0137       | 0.0134        | 0.0132       | 0.0132            | 0.0187     

License
===

This source code is public domain. You can do anything you want with it. It would be cool if you add attribution or just let me know that you used it for some project, but it's not required.