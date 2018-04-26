// Anisotropic Kuwahara Filtering on the GPU
// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "glview.h"
#include "mainwindow.h"
#include "glslmgr.h"


#ifndef M_PI
const double M_PI = 3.14159265358979323846;
#endif

GLView::GLView(QWidget *parent) 
    : QGLWidget(QGLFormat(QGL::NoDepthBuffer | QGL::DoubleBuffer), parent)
{
    m_glslMgr = new GLSLMgr(this);
    m_zoom = 1.0;
    m_drag = false;
    m_processN = 0;
    m_width = m_height = 0;
}

GLView::~GLView() {
}


void GLView::initializeGL() {
    if (!GLEE_VERSION_2_0 ||
        !GLEE_EXT_gpu_shader4 ||
        !GLEE_EXT_framebuffer_object ||
        !GLEE_ARB_texture_float ||
        !GLEE_ARB_texture_rectangle ||
        !GLEE_EXT_bgra) {
        QMessageBox::critical(this, "Error", 
            "OpenGL 2.0 Graphics Card with EXT_gpu_shader4, EXT_framebuffer_object, "
            "ARB_texture_rectangle, ARB_texture_float and EXT_bgra required!");
        exit(1);
    }

    if (!m_glslMgr->initialize()) exit(1);

    m_jetImage = QImage(":/jet.png");
    m_jet = bindTexture(m_jetImage);
    m_N = mainWindow->N->value();

    glEnable(GL_TEXTURE_2D);
    glGenTextures(TEX_MAX, m_tex);
    for (int i = 0; i < TEX_MAX; ++i) {
        glBindTexture(GL_TEXTURE_2D, m_tex[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffersEXT(1, &m_fbo);
    glClearColor(.8f, .8f, .8f, 1.0f);
    
    updateKernel();
}


void GLView::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-w/2, w-w/2, -h/2, h-h/2);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void GLView::paintGL() {
    int view = mainWindow->view->currentIndex();
    glClear(GL_COLOR_BUFFER_BIT);
    glPushMatrix();
    glTranslatef(m_origin.x(), -m_origin.y(), 0);
    glScalef(m_zoom, m_zoom, 1);
    glBindTexture(GL_TEXTURE_2D, m_tex[TEX_SRC + view]);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2i(-m_width/2, m_height-m_height/2);
    glTexCoord2f(0, 1);
    glVertex2i(-m_width/2, -m_height/2);
    glTexCoord2f(1, 1);
    glVertex2i(m_width-m_width/2, -m_height/2);
    glTexCoord2f(1, 0);
    glVertex2i(m_width-m_width/2, m_height-m_height/2);               
    glEnd();
    glPopMatrix();
}


void GLView::setPixels(int w, int h, GLenum format, GLenum type, void *pixels) {
    makeCurrent();
    if ((m_width != w) || (m_height != h)) {
        for (int i = TEX_SRC+1; i < TEX_MAX; ++i) {
            glBindTexture(GL_TEXTURE_2D, m_tex[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
        }

        float *noise = new float[w * h];
        {   
            float *p = noise;
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    *p++ = 0.5 + 1.25 * perlin_original_noise3(i/2.0, j/2.0, 0.5); 
                }
            }
        }
        glBindTexture(GL_TEXTURE_2D, m_tex[TEX_NOISE]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16F_ARB, w, h, 0, GL_LUMINANCE, GL_FLOAT, noise);
        assert(glGetError() == GL_NO_ERROR);
        delete[] noise;
    }

    m_width = w;
    m_height = h;

    glBindTexture(GL_TEXTURE_2D, m_tex[TEX_SRC]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, format, type, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);

    process();
}


void GLView::getPixels(int w, int h, GLenum format, GLenum type, void *pixels) {
    makeCurrent();
    int view = mainWindow->view->currentIndex();

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_tex[TEX_SRC + view], 0);
    glReadPixels(0, 0, w, h, format, type, pixels);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}


void GLView::mousePressEvent( QMouseEvent *e ) {
    if (e->buttons() == Qt::LeftButton) {
        m_drag = true;
        m_dragPos = e->pos();
        m_dragOrigin = m_origin;
    }
    updateGL();
}


void GLView::mouseReleaseEvent( QMouseEvent *e ) {
    if (m_drag && (e->button() == Qt::LeftButton)) {
        m_drag = false;
    }
    updateGL();
}


void GLView::mouseMoveEvent( QMouseEvent *e ) {
    if (m_drag) {
        m_origin = m_dragOrigin + e->pos() - m_dragPos;
        updateGL();
    }
}


void GLView::wheelEvent(QWheelEvent *e) {
    QSize sz = size();
    float u = e->delta() / 120.0 / 4.0;
    if (u < -0.5) u = -0.5;
    if (u > 0.5) u = 0.5;
    m_origin *= (1 + u);
    m_zoom *= (1 + u);
    updateGL();
}


void GLView::buildLog() {
    m_glslMgr->exec();
}


void GLView::zoomIn() {
    m_origin *= 2.0;
    m_zoom *= 2.0;
    updateGL();
}


void GLView::zoomOut() {
    m_origin *= 0.5;
    m_zoom *= 0.5;
    updateGL();
}


void GLView::reset() {
    m_zoom = 1.0;
    m_origin = QPoint(0, 0);
    updateGL();
}


static void gauss_filter(float *data, int width, int height, float sigma) {
    float twoSigma2 = 2.0 * sigma * sigma;
    int halfWidth = (int)ceil( 2.0 * sigma );

    float *src_data = new float[width * height];
    memcpy(src_data, data, width * height * sizeof(float));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0;
            float w = 0;

            for ( int i = -halfWidth; i <= halfWidth; ++i )	{
                for ( int j = -halfWidth; j <= halfWidth; ++j )	{
                    int xi = x + i;
                    int yj = y + j;
                    if ((xi >= 0) && (xi < width) && (yj >= 0) && (yj < height)) {
                        float r = sqrt((float)(i * i + j * j));
                        float k = exp( -r *r / twoSigma2 );
                        w += k;
                        sum += k * src_data[ xi + yj * width];
                    }
                }
            }

            data[ x + y * width ] = sum / w;
        }
    }

    delete[] src_data;
}


void make_sector(float *krnl, int k, int N, int size, float sigma_r, float sigma_s) {
    float *p = krnl;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            float x = i - 0.5 * size + 0.5;
            float y = j - 0.5 * size + 0.5;
            float r = sqrt((double)(x * x + y * y));

            float a = 0.5 * atan2(y, x) / M_PI + k * 1.0 / N;
            if (a > 0.5)
                a -= 1.0;
            if (a < -0.5)
                a += 1.0;

            if ((fabs(a) <= 0.5 / N) && (r < 0.5 * size)) {
                *p = 1;
            } else {
                *p = 0;
            }
            ++p;
        }
    }

    gauss_filter(krnl, size, size, sigma_s);

    p = krnl;
    float mx = 0.0;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            float x = i - 0.5 * size + 0.5;
            float y = j - 0.5 * size + 0.5;
            float r = sqrt((double)(x * x + y * y));
            *p *= exp(-0.5 * r * r / sigma_r / sigma_r);
            if (*p > mx) mx = *p;
            ++p;
        }
    }

    p = krnl;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            *p /= mx;
            ++p;
        }
    }
}


void GLView::updateKernel() {
    makeCurrent();

    int N = mainWindow->N->value();
    float smoothing = mainWindow->smoothing->value() / 100.0;

    const int krnl_size = 32;
    const float sigma = 0.25f * (krnl_size - 1);

    float *krnl[4];
    for (int k = 0; k < 4; ++k) {
        krnl[k] = new float[krnl_size * krnl_size];
        make_sector(krnl[k], k, N, krnl_size, sigma,  smoothing * sigma);
    }

    float *krnlx4 = new float[4 * krnl_size * krnl_size];
    for (int i = 0; i < krnl_size * krnl_size; ++i) {
        for (int k = 0; k < 4; ++k) {
            krnlx4[4*i+k] = krnl[k][i];
        }
    }

    glBindTexture(GL_TEXTURE_2D, m_tex[TEX_KRNL]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16F_ARB, krnl_size, krnl_size, 0, GL_LUMINANCE, GL_FLOAT, krnl[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glBindTexture(GL_TEXTURE_2D, m_tex[TEX_KRNLX4]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, krnl_size, krnl_size, 0, GL_RGBA, GL_FLOAT, krnlx4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glBindTexture(GL_TEXTURE_2D, 0);

    QImage img(krnl_size, krnl_size, QImage::Format_RGB32);
    for (int j = 0; j < krnl_size; ++j) {
        for (int i = 0; i < krnl_size; ++i) {
            int c = (int)(255.0 * krnl[0][j * krnl_size + i]);
            img.setPixel(i, j, m_jetImage.pixel(c, 0));
        }
    }
    mainWindow->kernel->setPixmap(QPixmap::fromImage(img));

    for (int k = 0; k < 4; ++k) {
        delete krnl[k];
    }
    delete krnlx4;
    process();
}


void bind_sampler(GLuint id, const char *name, GLint unit, GLuint texture) {
    GLuint location = glGetUniformLocation(id, name);
    assert(glGetError() == GL_NO_ERROR);
    glUniform1i(location, unit);
    assert(glGetError() == GL_NO_ERROR);
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, texture);
}


void GLView::process() {
    if (!m_width || !m_height) return;

    ++m_processN;
    #ifdef WIN32
    LARGE_INTEGER p_freq, p_start, p_stop;
    QueryPerformanceFrequency(&p_freq);
    QueryPerformanceCounter(&p_start);
    #endif

    int algorithm = mainWindow->algorithm->currentIndex();
    float sigma_t = mainWindow->sigma_t->value();
    float alpha = mainWindow->alpha->value();
    int N = mainWindow->N->value();
    int radius = mainWindow->radius->value();
    int q = mainWindow->q->value();

    if (N != m_N) {
        m_N = N;
        m_glslMgr->replaceInSource(QString("const int N = (\\d+);"), 
                                   QString("const int N = %1;").arg(N));
    }

    makeCurrent();
    glPushAttrib(GL_ENABLE_BIT | GL_VIEWPORT_BIT );
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);

    glViewport(0, 0, m_width, m_height);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    {
        GLuint pid = m_glslMgr->pid("sst");
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_tex[TEX_TMP0], 0);
        glUseProgram(pid);
        bind_sampler(pid, "src", 0, m_tex[TEX_SRC]);
        glRectf(-1,-1,1,1);

        pid = m_glslMgr->pid("gauss");
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_tex[TEX_TMP1], 0);
        glUseProgram(pid);
        bind_sampler(pid, "src", 0, m_tex[TEX_TMP0]);
        glUniform1f(glGetUniformLocation(pid, "sigma"), sigma_t);
        glRectf(-1,-1,1,1);

        pid = m_glslMgr->pid("tfm");
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_tex[TEX_TFM], 0);
        glUseProgram(pid);
        bind_sampler(pid, "src", 0, m_tex[TEX_TMP1]);
        glRectf(-1,-1,1,1);

        pid = m_glslMgr->pid("lic");
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_tex[TEX_LIC], 0);
        glUseProgram(pid);
        bind_sampler(pid, "tfm", 0, m_tex[TEX_TFM]);
        bind_sampler(pid, "src", 1, m_tex[TEX_NOISE]);
        glUniform1f(glGetUniformLocation(pid, "sigma"), 3);
        glRectf(-1,-1,1,1);

        pid = m_glslMgr->pid("showa");
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_tex[TEX_A], 0);
        glUseProgram(pid);
        bind_sampler(pid, "tfm", 0, m_tex[TEX_TFM]);
        bind_sampler(pid, "jet", 1, m_jet);
        glRectf(-1,-1,1,1);

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_tex[TEX_DST], 0);
        if (algorithm == 0) {
            pid = m_glslMgr->pid("kuwahara");
            glUseProgram(pid);
            bind_sampler(pid, "img", 0, m_tex[TEX_SRC]);
            glUniform1i(glGetUniformLocation(pid, "radius"), (int)radius);
        } else if (algorithm == 1) {
            GLuint pid = m_glslMgr->pid("gkf");
            glUseProgram(pid);
            bind_sampler(pid, "src", 0, m_tex[TEX_SRC]);
            bind_sampler(pid, "K0", 1, m_tex[TEX_KRNL]);
            glUniform1i(glGetUniformLocation(pid, "radius"), (int)radius);
            glUniform1f(glGetUniformLocation(pid, "q"), q);
        } else if (algorithm == 2) {
            GLuint pid = m_glslMgr->pid("akf_v1");
            glUseProgram(pid);
            bind_sampler(pid, "src", 0, m_tex[TEX_SRC]);
            bind_sampler(pid, "K0", 1, m_tex[TEX_KRNL]);
            bind_sampler(pid, "tfm", 2, m_tex[TEX_TFM]);
            glUniform1f(glGetUniformLocation(pid, "alpha"), alpha);
            glUniform1f(glGetUniformLocation(pid, "radius"), radius);
            glUniform1f(glGetUniformLocation(pid, "q"), q);
        } else {
            GLuint pid = m_glslMgr->pid("akf_v2");
            glUseProgram(pid);
            bind_sampler(pid, "src", 0, m_tex[TEX_SRC]);
            bind_sampler(pid, "K0123", 1, m_tex[TEX_KRNLX4]);
            bind_sampler(pid, "tfm", 2, m_tex[TEX_TFM]);
            glUniform1f(glGetUniformLocation(pid, "alpha"), alpha);
            glUniform1f(glGetUniformLocation(pid, "radius"), radius);
            glUniform1f(glGetUniformLocation(pid, "q"), q);
        }
      
        if (mainWindow->renderFullQuad->isChecked()) {
            glRectf(-1,-1,1,1);
        } else {
            int delta = 8192 / m_width;
            int y1 = 0;
            int y2 = 0;
            while (y2 < m_height) {
                y1 = y2;
                y2 = y1 + delta;
                if (y2 > m_height)
                    y2 = m_height;
                glRectf(-1, -1 + 2.0 * y1 / m_height, 1, -1 + 2.0 * y2 / m_height);
            }
        }
    }

    glUseProgram(0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPopAttrib();
    glFinish();

    #ifdef WIN32
    QueryPerformanceCounter(&p_stop);
    double s = (double)(p_stop.QuadPart - p_start.QuadPart) / p_freq.QuadPart;
    mainWindow->pf->setText(QString("%1 ms / %2 fps / %3").arg(s * 1000, 0, 'g', 5)
                                                          .arg(1.0 / s, 0, 'g', 3)
                                                          .arg(m_processN));
    #endif WIN32

    updateGL();
}
