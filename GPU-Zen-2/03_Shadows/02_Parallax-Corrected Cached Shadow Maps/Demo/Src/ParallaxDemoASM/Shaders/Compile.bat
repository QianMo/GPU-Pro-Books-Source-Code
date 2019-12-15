@echo off
fxc ColorPass.shader /Tps_5_0 /EmainPS /DSHADOWS_CROSSFADE=1 /DASM_MRF=1 /DASM_PRERENDER_AVAILABLE=1 /DASM_PCF9=1
rem fxc _Shaders\MeshDepthOnly.shader /Tps_5_0 /EmainPS /DASM_LAYER=1
rem fxc _Shaders\MeshDepthOnly.shader /Tvs_5_0 /EmainVS /DASM_LAYER=1
