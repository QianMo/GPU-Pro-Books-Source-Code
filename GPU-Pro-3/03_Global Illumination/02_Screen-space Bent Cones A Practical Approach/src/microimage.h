#ifndef ML_GLUT_DEMO_MICROIMAGE_H
#define ML_GLUT_DEMO_MICROIMAGE_H

class MicroImage {
public:
    MicroImage();
    MicroImage(unsigned int w, unsigned int h);
    ~MicroImage();

    bool loadBMPFromMemory(const unsigned char *data, unsigned int size, bool flip = false);
    bool loadBMPFromFile(const char * name, bool flip = false);

    unsigned int width() const { return width_;}
    unsigned int height() const { return height_;}
    unsigned int layers() const { return 3;}

    unsigned char *data() { return data_; }
    unsigned char *pixel(unsigned int x, unsigned int y);

private:
    unsigned int width_, height_;
    unsigned char *data_;
};

#endif