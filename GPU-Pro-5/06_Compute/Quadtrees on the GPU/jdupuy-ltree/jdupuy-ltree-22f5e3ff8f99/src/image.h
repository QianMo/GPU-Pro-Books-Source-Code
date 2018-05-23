// image.h - public domain APNG loader / exporter utility for OpenGL
// author: Jonathan Dupuy (jdupuy at liris.cnrs.fr)
#ifndef IMAGE_H
#define IMAGE_H

// Error codes
enum {IMAGE_FAILURE = 0, IMAGE_SUCCESS};

// Loaders (return NULL if error)
struct image_t *image_load_memory (int width, int height, int depth,
                                    GLenum glformat, GLenum gltype,
                                    const GLvoid *data);
#if 1 
struct image_t *image_load_png (const char *filename);
#else // TODO
struct image_t *image_load_apng (const char *filename);
#endif
struct image_t *image_load_gl_texture (GLenum target, int level);
struct image_t *image_load_gl_texture_formatted (GLenum target, int level,
                                                  GLenum format, GLenum type);
struct image_t *image_load_gl_colorbuffer (GLenum glbuffer);
struct image_t *image_load_gl_colorbuffer_formatted (GLenum buffer,
                                                      GLenum format,
                                                      GLenum type);

// Destructor
void image_release (struct image_t *image);

// Accessors
int image_get_width (const struct image_t *image);
int image_get_height (const struct image_t *image);
int image_get_depth (const struct image_t *image);
GLenum image_get_gl_format (const struct image_t *image);
GLenum image_get_gl_type (const struct image_t *image);
const void *image_get_data (const struct image_t *image);

// OpenGL texture loaders
int image_glTexImage1D (GLenum target, GLint level, GLint internalFormat,
                        GLboolean immutable, GLboolean mipmap,
                        const struct image_t *data);
int image_glTexImage2D (GLenum target, GLint level, GLint internalFormat,
                        GLboolean immutable, GLboolean mipmap,
                        const struct image_t *data);
int image_glTexImage3D (GLenum target, GLint level, GLint internalFormat,
                        GLboolean immutable, GLboolean mipmap,
                        const struct image_t *data);
int image_glTexSubImage1D (GLenum target, GLint level,
                           GLint xoffset,
                           const struct image_t *data);
int image_glTexSubImage2D (GLenum target, GLint level,
                           GLint xoffset, GLint yoffset,
                           const struct image_t *data);
int image_glTexSubImage3D (GLenum target, GLint level,
                           GLint xoffset, GLint yoffset, GLint zoffset,
                           const struct image_t *data);

// Exports
int image_save_gl_front_buffer (void);
int image_save_gl_front_buffer_as (const char *filename);
int image_save_png (const struct image_t *image, const char *filename);
#if 0
int image_save_apng (const struct image_t *image, const char *filename); // TODO
#endif

#endif
