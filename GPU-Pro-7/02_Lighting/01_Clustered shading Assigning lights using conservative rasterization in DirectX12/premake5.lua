-- premake5.lua
workspace "DX12_ConservativeClustered"
    configurations { "Debug", "Release" }
    platforms { "x64" }
	startproject "DX12_ConservativeClustered"
	
	location "build/"

project "DX12_ConservativeClustered"
    kind        "ConsoleApp"
    language    "C++"
    targetdir   "bin/"
    files       { "source/DX12_ConservativeClustered/**.h", "source/DX12_ConservativeClustered/**.cpp" }
    includedirs { "external/SDL/include/", "external/DirectXTK/include/", "external/AntTweakBar/include/" }
     
    filter "configurations:Debug"
        links       { "external/SDL/lib/x64/SDL2", "external/SDL/lib/x64/SDL2_image", "external/DirectXTK/lib/x64/Debug/DirectXTK", "d3d12", "dxgi", "d3dcompiler", "external/AntTweakBar/lib/AntTweakBar64" }
        objdir      "%{wks.location}/%{prj.name}/debug/obj/"
        targetsuffix "_debug"
        flags       { "Symbols", "FatalCompileWarnings", "FatalLinkWarnings", "MultiProcessorCompile", "Unicode" }
        
    filter "configurations:Release"
        links       { "external/SDL/lib/x64/SDL2", "external/SDL/lib/x64/SDL2_image", "external/DirectXTK/lib/x64/Release/DirectXTK", "d3d12", "dxgi", "d3dcompiler", "external/AntTweakBar/lib/AntTweakBar64"  }
        objdir      "%{wks.location}/%{prj.name}/release/obj/"
        optimize    "On"
        targetsuffix "_release"
        flags       { "FatalCompileWarnings", "FatalLinkWarnings", "MultiProcessorCompile", "Unicode" }