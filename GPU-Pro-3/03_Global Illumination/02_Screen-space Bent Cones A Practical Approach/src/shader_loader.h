#pragma once

#include <string>
#include <map>
#include <set>

class ShaderLoader {
public:
    static std::string loadShader(const std::string& file);
	static void setFilesAsSource(bool value) { InstanceRef().liveUpdate_ = value; };

private:
    static inline ShaderLoader* Instance() {
        if(!instance_)
            instance_ = new ShaderLoader();
        instance_->init();
        return instance_;
    }
    static inline ShaderLoader& InstanceRef() {
        return *Instance();
    }

    ShaderLoader() : liveUpdate_(false) {};
    ShaderLoader(const ShaderLoader&);
    ~ShaderLoader();
    ShaderLoader& operator=(const ShaderLoader&);

    static ShaderLoader* instance_;

private:
	void init();
	std::string addShaderToCompilation(const std::string& file);

private:
    bool liveUpdate_;
	std::set<std::string> writtenFilesThisSession_;
    static std::map<std::string, std::string> fileName2Source_;
};