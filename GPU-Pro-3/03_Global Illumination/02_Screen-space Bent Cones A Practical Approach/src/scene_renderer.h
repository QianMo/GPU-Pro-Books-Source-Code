#pragma once

#include <nvModel.h>
#include "gbuffer.h"

class SceneRenderer {
public:
    void init(const std::string& modelDirectory, const std::string& modelFile, bool flipYZ = false, float scale = 1.0f);
    void cleanup();

    void renderGeometry(GBuffer& gbuffer) const;

private:
    void buildRenderModel();

private:
    struct TexturedPart {
        GLuint vao;
        GLuint vBuf;
        int vCount;
        GLuint map_kd;
        GLuint map_ks;
        float ka[3], kd[3], ks[3], spec;
    };

private:
	bool flipYZ_;
	float scale_;
	std::string modelDirectory_;
	std::string modelFile_;

    nv::Model nvModel_;

    std::vector<TexturedPart> model_;

    float maxAniso;
};
