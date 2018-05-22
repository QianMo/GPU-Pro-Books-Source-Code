#include "shader_loader.h"

#include <iostream>
#include <fstream>

namespace {
    std::string loadFile(const std::string& Filename) {
        std::ifstream stream(Filename.c_str(), std::ios::in);

        if(!stream.is_open())
            return "";

        std::string Line = "";
        std::string Text = "";

        while(getline(stream, Line))
            Text += Line + "\n";

        stream.close();

        return Text;
    }

	bool fileExists(const std::string& filename) {
		bool flag = false;
		std::fstream fin;
		fin.open(filename.c_str(), std::ios::in);
		if(fin.is_open()) {
			flag = true;
		}
		fin.close();

		return flag;
	}
}
    
ShaderLoader* ShaderLoader::instance_ = NULL;
std::map<std::string, std::string> ShaderLoader::fileName2Source_;

void ShaderLoader::init() {
#include "shaderCollection.shader"
}

std::string ShaderLoader::loadShader(const std::string& file) {
    ShaderLoader& this_ = InstanceRef();
	if(!this_.liveUpdate_ && this_.fileName2Source_.find(file) != this_.fileName2Source_.end()) return this_.fileName2Source_[file];
	if(!fileExists(file)) return this_.fileName2Source_[file];

	return this_.addShaderToCompilation(file);
}

std::string ShaderLoader::addShaderToCompilation(const std::string& file) {
	static bool firstWrite = true;
	if(firstWrite) {
		// clear file
		std::ofstream ofile("./src/shaderCollection.shader", std::ios_base::trunc);
		ofile.flush();
		ofile.close();
		fileName2Source_.clear();
	}
	firstWrite = false;

	if(fileName2Source_.find(file) == fileName2Source_.end()) {
		std::cout << "Adding file \"" << file << "\" to compilation" << std::endl;
		std::ofstream ofile("./src/shaderCollection.shader", std::ios_base::app);
		ofile << std::endl << "fileName2Source_[\"" << file << "\"] = " << std::endl;
		std::ifstream stream(file.c_str(), std::ios::in);

		std::string Line = "";
		size_t pos = 0;

		while(getline(stream, Line)) {
			while(true) {
				pos = Line.find("\"", pos+2);
				if(pos == std::string::npos) break;
				Line.replace(pos, 1, "\\\"");
			}
			ofile << "\"" << Line << " \\n\"" << std::endl;
		}
		ofile << "\"\";" << std::endl;
		ofile << "//////////////////////////////////////////////////////////////////////////" << std::endl;
		ofile << "//////////////////////////////////////////////////////////////////////////" << std::endl;
		ofile << "//////////////////////////////////////////////////////////////////////////" << std::endl;

		stream.close();
		ofile.close();

		fileName2Source_[file] = loadFile(file);
	}
	return fileName2Source_[file];
}
