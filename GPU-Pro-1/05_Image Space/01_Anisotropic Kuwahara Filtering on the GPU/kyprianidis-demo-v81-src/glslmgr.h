// Anisotropic Kuwahara Filtering on the GPU
// by Jan Eric Kyprianidis <www.kyprianidis.com>
#ifndef INCLUDED_GLSLMGR_H
#define INCLUDED_GLSLMGR_H

#include "ui_glslmgr.h"

class GLSLMgr : public QDialog, public Ui_GLSLMgr {
    Q_OBJECT
public:
	GLSLMgr(QGLWidget *parent);
    ~GLSLMgr();

    QGLWidget* parent() { return (QGLWidget*)QDialog::parent(); }
    GLuint pid(const char* name) { return m_pid[name]; }

    bool initialize();
	void replaceInSource(const QString& pattern, const QString& text);

public slots:
    void build();
    void enableEdit();
    void sourceChanged(bool change);
    void copyToClipboard();

private:
    QShortcut *m_enableEdit;
    int m_buildStatus;
    QString m_log;
    QTextEdit *m_logText;
    QList<QPlainTextEdit*> m_srcText;
    QPushButton *m_build;
	QMap<QString, GLuint> m_pid;
};

#endif
