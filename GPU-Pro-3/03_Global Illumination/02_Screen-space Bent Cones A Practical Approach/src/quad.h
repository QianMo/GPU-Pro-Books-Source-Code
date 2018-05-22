#pragma once

#include "rendergeometry.h"

class Quad : public RenderGeometry {
public:
    static inline Quad* Instance() {
        if(!instance_)
            instance_ = new Quad();
        return instance_;
    }
    static inline Quad& InstanceRef() {
        return *Instance();
    }

    virtual void init();
    virtual void cleanup();

protected:
    Quad();

    static const GLsizei numVertices_ = 4;

private:
    Quad(const Quad&);
    ~Quad();
    Quad& operator=(const Quad&);

    static Quad* instance_;
};

