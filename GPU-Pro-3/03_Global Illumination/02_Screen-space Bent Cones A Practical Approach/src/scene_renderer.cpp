#include "scene_renderer.h"
#include "config.h"
#include <string>
#include <iostream>

#include "microimage.h"

namespace {
    //const std::string modelFile = "_urban-level-01-big-3ds_notrees.obj";
    //const std::string modelFile = "_urban-level-01-big_EDIT.obj";
    const std::string modelFile = "urban-level-01-big_edit.obj";
    const std::string modelDir = "models_bmp\\";

    bool loadTexResource(GLenum target, const std::string& path, bool flip = false) {
        MicroImage mImage;
        if(mImage.loadBMPFromFile(path.c_str(), flip)) {
			GLCHECK(glTexImage2D(target, 0, GL_RGB, mImage.width(), mImage.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, mImage.data()));
            return true;
        } else {
            return false;
        }
    }

    struct Vertex {
        Vertex(nv::vec3f const & position, nv::vec3f const & normal, nv::vec2f const & texcoord) 
            : position_(position), normal_(normal), texcoord_(texcoord) {}

        nv::vec3f position_;
        nv::vec3f normal_;
        nv::vec2f texcoord_;
    };
}

void SceneRenderer::init(const std::string& modelDirectory, const std::string& modelFile, bool flipYZ, float scale) {
	modelDirectory_ = modelDirectory;
	modelFile_ = modelFile;

	flipYZ_ = flipYZ;
	scale_ = scale;

    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);

	if(!nvModel_.loadModelFromFile((modelDirectory_+modelFile_).c_str())) {
		std::cerr << "model_ loading failed!" << std::endl;
		exit(1);
	}

	nvModel_.compileModel();

	buildRenderModel();
}

void SceneRenderer::cleanup() {

}

void SceneRenderer::buildRenderModel() {
    std::vector<std::vector<Vertex>> vertices(nvModel_.getMaterialCount());

    // sort by material
    const float *cData = nvModel_.getCompiledVertices();
    const int *mData = nvModel_.getMaterialIndices();
    nv::vec3f v, n;
    nv::vec2f t;
    const GLuint *idx = nvModel_.getCompiledIndices();
    for(int i = 0; i<nvModel_.getCompiledIndexCount(); ++i) {
        const float *currC = cData + idx[i]*nvModel_.getCompiledVertexSize() + nvModel_.getCompiledPositionOffset();
        v[0] = *(currC++);
        v[1] = *(currC++);
        v[2] = *(currC++);
        //if (nvModel_.getPositionSize()==4) v[3] = *(currC++); else v[3] = 1.0f; 
        currC = cData + idx[i]*nvModel_.getCompiledVertexSize() + nvModel_.getCompiledNormalOffset();
        n[0] = *(currC++);
        n[1] = *(currC++);
        n[2] = *(currC++);
        //n[3] = 0.0f;
        currC = cData + idx[i]*nvModel_.getCompiledVertexSize() + nvModel_.getCompiledTexCoordOffset();
        t[0] = *(currC++);
        t[1] = *(currC++);
        //v[3] = t[0];
        //n[3] = t[1];
        int mat = *(mData + i);
		v *= scale_;
		if(flipYZ_) std::swap(n[1], n[2]);
		if(flipYZ_) std::swap(v[1], v[2]);
        vertices[mat].push_back(Vertex(v, n, t));
    }

    // move to opengl
    model_.clear();
    model_.resize(nvModel_.getMaterialCount());
    //model_.pop_back();
    for(size_t i=0; i<model_.size(); ++i) {
        TexturedPart &tp = model_[i];

        tp.vCount = vertices[i].size();

        // create buffers and fill with data
        GLCHECK(glGenBuffers(1, &tp.vBuf));
        GLCHECK(glBindBuffer(GL_ARRAY_BUFFER, tp.vBuf));
        GLCHECK(glBufferData(GL_ARRAY_BUFFER, tp.vCount*sizeof(Vertex), &(vertices[i][0]), GL_STATIC_DRAW));
        GLCHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

        glGenVertexArrays(1, &tp.vao);
        glBindVertexArray(tp.vao);
        GLCHECK(glBindBuffer(GL_ARRAY_BUFFER, tp.vBuf));
        GLCHECK(glVertexAttribPointer(vertexAttrib::POSITION, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), GLH_BUFFER_OFFSET(0)));
        GLCHECK(glVertexAttribPointer(vertexAttrib::NORMAL, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), GLH_BUFFER_OFFSET(sizeof(nv::vec3f))));
        GLCHECK(glVertexAttribPointer(vertexAttrib::TEXCOORD, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), GLH_BUFFER_OFFSET(sizeof(nv::vec3f)*2)));
        GLCHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

        glEnableVertexAttribArray(vertexAttrib::POSITION);
        glEnableVertexAttribArray(vertexAttrib::NORMAL);
        glEnableVertexAttribArray(vertexAttrib::TEXCOORD);
        glBindVertexArray(0);

        nv::Model::Material* mat = const_cast<nv::Model::Material*>(nvModel_.getMaterial(i));

		// copy material
		tp.ka[0] = mat->ka[0]; tp.ka[1] = mat->ka[1]; tp.ka[2] = mat->ka[2];
		tp.kd[0] = mat->kd[0]; tp.kd[1] = mat->kd[1]; tp.kd[2] = mat->kd[2];
		tp.ks[0] = mat->ks[0]; tp.ks[1] = mat->ks[1]; tp.ks[2] = mat->ks[2];
		tp.spec = mat->ns;

        // load texture
        if(mat->map_kd.empty()) {
            mat->map_kd = std::string("white.bmp");
        }
		else {
			tp.kd[0] = 1.0f;
			tp.kd[1] = 1.0f;
			tp.kd[2] = 1.0f;
		}
        GLCHECK(glGenTextures(1, &tp.map_kd));
        GLCHECK(glBindTexture(GL_TEXTURE_2D, tp.map_kd));
        if(!loadTexResource(GL_TEXTURE_2D, modelDirectory_+mat->map_kd, false)) {
            GLCHECK(glDeleteTextures(1, &tp.map_kd));
            tp.map_kd = 0;
        } else {
            GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
            GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
            GLCHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
            GLCHECK(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAniso));

			glGenerateMipmap(GL_TEXTURE_2D);
        }
		GLCHECK(glBindTexture(GL_TEXTURE_2D, 0));
	}
}

void SceneRenderer::renderGeometry(GBuffer& gbuffer) const {
	GLCHECK(glActiveTexture(GL_TEXTURE0));

	for(size_t j=0; j<model_.size(); ++j) {
		const TexturedPart& tp = model_[j];

		// material (diffuse)
		GLCHECK(glBindTexture(GL_TEXTURE_2D, tp.map_kd));

		gbuffer.setMaterialParameter(tp.kd);

		GLCHECK(glBindVertexArray(tp.vao));
		GLCHECK(glDrawArrays(GL_TRIANGLES, 0, tp.vCount));
	}
}
