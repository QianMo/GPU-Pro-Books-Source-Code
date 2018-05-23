-- ---------------------------------------------------------
solution "opengl"
	configurations {"debug", "release"}
	location       ( _OPTIONS["to"] )
		includedirs {"../lib/"}

		language    "C"
		kind        "ConsoleApp"
		files       { "../src/*.c" }
		defines     {"GLFW_NO_GLU"} -- lib specific

-- Debug configuration
	configuration   {"debug"}
--		targetname  "demo_debug"
		targetdir   (_OPTIONS["to"])
		defines     {"DEBUG"}
		flags       {"ExtraWarnings",
		             "EnableSSE",
		             "EnableSSE2",
		             "Symbols"}

-- Release configuration
	configuration   {"release"}
--		targetname  "demo"
		targetdir   (_OPTIONS["to"])
		defines     {"NDEBUG"}
		flags       {"EnableSSE",
		             "EnableSSE2",
		             "Optimize"}

-- unix specific
	configuration   {"linux"}
		links       {"m", "png", "glfw3", "GL"}
		links       {"Xrandr", "Xxf86vm", "Xi", "pthread", "X11", "rt"}
		defines     {"LINUX"}

-- macosx specific (untested)
--	configuration    "macosx"
--			defines  {"LUA_USE_MACOSX"}

--	configuration    {"macosx", "gmake"}
--		buildoptions {"-mmacosx-version-min=10.1"}
--		linkoptions  {"-lstdc++-static", "-mmacosx-version-min=10.1"}

-- windows specific
	configuration    "windows"
		includedirs {"../lib/libpng"}
		includedirs {"../lib/GLFW"}
		defines     {"WINDOWS"}
		links       {"../lib/GLFW/libglfw3.a"} 
		links       {"../lib/libpng/zlib.lib", "../lib/libpng/libpng14.lib"}
		links       {"opengl32"}
		links       {"gdi32", "user32", "kernel32"}

-- ---------------------------------------------------------
-- Project 

	project "quadtree_simple"
		excludes    {"../src/octree_simple.c"}
		excludes    {"../src/quadtree_simple.c"}

--	project "octree_simple"

	project "terrain"
		excludes    {"../src/octree_simple.c"}
		excludes    {"../src/quadtree_simple.c"}
		defines     {"TERRAIN_RENDERER"}

	project "parametric"
		excludes    {"../src/quadtree_simple.c"}
		excludes    {"../src/octree_simple.c"}
		defines     {"PARAMETRIC_RENDERER"}

-- ---------------------------------------------------------	
-- cleanup
	if _ACTION == "clean" then
		os.rmdir("../bin")
		os.rmdir("../build")
	end

-- copy dlls for windows
-- for some reason, I have to call release.lua twice on windows
-- so that the dlls are effectively copied...
	if _OPTIONS["os"] == "windows" then
		os.copyfile("../lib/libpng/libpng14-14.dll", _OPTIONS["to"].."/libpng14-14.dll")
		os.copyfile("../lib/GLFW/glfw3.dll", _OPTIONS["to"].."/glfw3.dll")
	end

--
-- Use the --to=path option to control where the project files get generated. I use
-- this to create project files for each supported toolset, each in their own folder,
-- in preparation for deployment.
--
	newoption {
		trigger = "to",
		value   = "path",
		description = "Set the output location for the generated files"
	}

