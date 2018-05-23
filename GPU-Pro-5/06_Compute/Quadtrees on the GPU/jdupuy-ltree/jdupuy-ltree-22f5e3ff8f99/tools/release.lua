

local function exec(cmd, ...)
	cmd = string.format(cmd, unpack(arg))
	local z = os.execute(cmd .. " > output.log 2> error.log")
	os.remove("output.log")
	os.remove("error.log")
	return z
end

print("Generating project files...")

--exec("premake4 /to=../build/vs2005 vs2005")
--exec("premake4 /to=../build/vs2008 vs2008")
--exec("premake4 /to=../build/vs2010 vs2010")
exec("premake4 /to=../build/gmake.unix /os=linux gmake")
exec("premake4 /to=../build/gmake.windows /os=windows gmake")
--exec("premake4 /to=../build/gmake.macosx /os=macosx /platform=universal32 gmake")
exec("premake4 /to=../build/codelite.unix /os=linux codelite")
exec("premake4 /to=../build/codelite.windows /os=windows codelite")
--exec("premake4 /to=../build/codelite.macosx /os=macosx /platform=universal32 codelite")
exec("premake4 /to=../build/codeblocks.unix /os=linux codeblocks")
exec("premake4 /to=../build/codeblocks.windows /os=windows codeblocks")
--exec("premake4 /to=../build/codeblocks.macosx /os=macosx /platform=universal32 codeblocks")
--exec("premake4 /to=../build/xcode3 /platform=universal32 xcode3")

print("Done.")

