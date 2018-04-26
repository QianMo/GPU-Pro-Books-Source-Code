// Anisotropic Kuwahara Filtering on the GPU
// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "glslmgr.h"


class Highlighter : public QSyntaxHighlighter {
public:
    Highlighter(QTextDocument *parent) : QSyntaxHighlighter(parent) {
	    static QStringList keywords = (QStringList()
            << "attribute" << "const" << "uniform" << "varying" 
            << "centroid" 
            << "break" << "continue" << "do" << "for" << "while"
            << "if" << "else"
            << "in" << "out" << "inout"
            << "float" << "int" << "void" << "bool" << "true" << "false"
            << "invariant"
            << "discard" << "return"
            << "mat2" << "mat3" << "mat4"
            << "mat2x2" << "mat2x3" << "mat2x4"
            << "mat3x2" << "mat3x3" << "mat3x4"
            << "mat4x2" << "mat4x3" << "mat4x4"
            << "vec2" << "vec3" << "vec4" << "ivec2" << "ivec3" << "ivec4" << "bvec2" << "bvec3" << "bvec4"
            << "sampler1D" << "sampler2D" << "sampler3D" << "samplerCube"
            << "sampler1DShadow" << "sampler2DShadow"
            << "struct"
        );
	    static QStringList functions = (QStringList()
            << "sin" << "cos" << "tan" 
            << "asin" << "acos" << "atan"
            << "radians" << "degrees" 
            << "pow" << "exp" << "log" << "exp2" << "log2" 
            << "sqrt" << "inversesqrt" 
            << "abs" << "ceil" << "clamp" << "floor" << "fract" 
            << "max" << "min" << "mix" << "mod" << "sign"
            << "smoothstep" << "step" << "ftransform" << "cross" 
            << "distance" << "dot" << "faceforward" 
            << "length" << "normalize" << "reflect" << "refract" 
            << "dFdx" << "dFdy" << "fwidth" << "matrixCompMult"
            << "all" << "any" << "equal" << "greaterThan" << "greaterThanEqual"
            << "lessThen" << "lessThenEqual" << "not" << "notEqual" << "texture1D"
            << "texture1DProj" << "texture2D" << "texture2DProj" << "texture3D"
            << "texture3DProj" << "textureCube" << "shadow1D" << "shadow1DProj"
            << "shadow2D" << "shadow2DProj" 
            << "shadow1DLod" << "shadow2DLodProj" << "shadow1DLod" << "shadow2DLodProj" 
            << "noise1" << "noise2" << "noise3" << "noise4"
         );

         Rule rule;
         m_keywordFormat.setForeground(Qt::blue);
         foreach (const QString &keyword, keywords) {
             rule.pattern = QRegExp(QString("\\b") + keyword + "\\b");
             rule.format = m_keywordFormat;
             m_rules.append(rule);
         }

         m_functionFormat.setForeground(Qt::darkRed);
         foreach (const QString &function, functions) {
             rule.pattern = QRegExp(QString("\\b") + function + "\\b");
             rule.format = m_functionFormat;
             m_rules.append(rule);
         }

         m_commentFormat.setForeground(Qt::darkGreen);
         rule.pattern = QRegExp("//[^\n]*");
         rule.format = m_commentFormat;
         m_rules.append(rule);

         m_preprocessorFormat.setForeground(Qt::darkMagenta);
         rule.pattern = QRegExp("#[^\n]*");
         rule.format = m_preprocessorFormat;
         m_rules.append(rule);
                                                   
         m_quotationFormat.setForeground(Qt::darkYellow);
         rule.pattern = QRegExp("\".*\"");
         rule.format = m_quotationFormat;
         m_rules.append(rule);

         m_commentStartExpression = QRegExp("/\\*");
         m_commentEndExpression = QRegExp("\\*/");
     }

     
     void highlightBlock(const QString &text) {
         foreach (const Rule &rule, m_rules) {
             QRegExp expression(rule.pattern);
             int index = expression.indexIn(text);
             while (index >= 0) {
                 int length = expression.matchedLength();
                 setFormat(index, length, rule.format);
                 index = expression.indexIn(text, index + length);
             }
         }
         setCurrentBlockState(0);

         int startIndex = 0;
         if (previousBlockState() != 1)
             startIndex = m_commentStartExpression.indexIn(text);

         while (startIndex >= 0) {
             int endIndex = m_commentEndExpression.indexIn(text, startIndex);
             int commentLength;
             if (endIndex == -1) {
                 setCurrentBlockState(1);
                 commentLength = text.length() - startIndex;
             } else {
                 commentLength = endIndex - startIndex
                                 + m_commentEndExpression.matchedLength();
             }
             setFormat(startIndex, commentLength, m_commentFormat);
             startIndex = m_commentStartExpression.indexIn(text, startIndex + commentLength);
         }
     }

private:
    struct Rule {
        QRegExp pattern;
        QTextCharFormat format;
    };
    QVector<Rule> m_rules;

    QRegExp m_commentStartExpression;
    QRegExp m_commentEndExpression;

    QTextCharFormat m_keywordFormat;
    QTextCharFormat m_functionFormat;
    QTextCharFormat m_preprocessorFormat;
    QTextCharFormat m_commentFormat;
    QTextCharFormat m_quotationFormat;
};


class SourceEdit : public QPlainTextEdit {
public:
    class LineNumbers : public QWidget {
    public:
        LineNumbers( SourceEdit *parent ) : QWidget(parent) {
            m_edit = parent;
            QFont font("Courier");
            font.setPixelSize(9);
            setFont(font);
            setFixedWidth( fontMetrics().width( QString("000"))+12 );

            QPalette p = palette();
            p.setColor(backgroundRole(), QColor(0xF0, 0xF0, 0xF0));
            setPalette(p);

            connect(m_edit->document()->documentLayout(), SIGNAL(update(const QRectF&)), this, SLOT(update()));
            connect(m_edit, SIGNAL(updateRequest(const QRect &, int)), this, SLOT(update()));
        }

        void paintEvent(QPaintEvent *event) {
            QRect rect = event->rect();
            QPainter p(this);
            p.eraseRect(rect);

            QTextBlock block = m_edit->firstVisibleBlock();
            qreal top = m_edit->blockBoundingGeometry(block).translated(m_edit->contentOffset()).top();
            qreal bottom = top + m_edit->blockBoundingRect(block).height();
            int line = block.blockNumber() + 1;

            while (block.isValid() && top <= rect.bottom()) {
                if (block.isVisible() && bottom >= rect.top()) {
                    p.drawText(QRectF(0, top, width()-6, bottom - top), 
                        Qt::AlignVCenter | Qt::AlignRight, QString::number(line));
                }
                block = block.next();
                top = bottom;
                bottom = top + m_edit->blockBoundingRect(block).height();
                ++line;
            }
        }
    private:
        SourceEdit *m_edit;
    };

    SourceEdit(QWidget *parent = NULL) : QPlainTextEdit(parent) {
        m_lineNumbers = new LineNumbers(this);
        QFont font("Courier");
        font.setPixelSize(11);
        setFont(font);
        setFrameShape(QFrame::NoFrame);
        setViewportMargins(m_lineNumbers->width(), 0, 0, 0);
    }

    void resizeEvent(QResizeEvent *e) {
        QPlainTextEdit::resizeEvent(e);
        QRect cr = contentsRect();
        m_lineNumbers->setGeometry(QRect(cr.left(), cr.top(), m_lineNumbers->width(), cr.height()));
    }

    friend class LineNumbers;
    LineNumbers *m_lineNumbers;
};


GLSLMgr::GLSLMgr(QGLWidget *parent) : QDialog(parent) {
    setAttribute(Qt::WA_DeleteOnClose, false); 
    setupUi(this);

    QSettings settings;
    m_showWarnings->setChecked(settings.value("showWarnings", true).toBool());

    m_enableEdit = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_E), this, SLOT(enableEdit()));
    connect(m_copy, SIGNAL(clicked()), this, SLOT(copyToClipboard()));

    m_tab->clear();
    m_logText = new QTextEdit;
    QFont font("Courier");
    font.setPixelSize(10);
    m_logText->setFont(font);
    m_logText->setFrameShape(QFrame::NoFrame);
    m_tab->addTab(m_logText, "Log");

	QDir glsl_dir(":/glsl");
	QFileInfoList glsl_list = glsl_dir.entryInfoList();
	for (int i = 0; i < glsl_list.size(); ++i) {
		QFileInfo fi = glsl_list[i];
		QFile f(fi.filePath());
		if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
			QMessageBox::critical(NULL, "Error", QString("Can't open %1").arg(fi.filePath()));
			qApp->quit();
		}
     
        QByteArray src = f.readAll();
        SourceEdit *code = new SourceEdit;

        new Highlighter(code->document());
        code->setPlainText(src);
        code->setDocumentTitle(fi.baseName());
        code->document()->setModified(false);
        m_srcText.append(code);
        connect(code->document(), SIGNAL(modificationChanged(bool)), this, SLOT(sourceChanged(bool)));
    }

    m_build = m_buttonBox->addButton("Build", QDialogButtonBox::ActionRole);
    m_build->hide();
    connect(m_build, SIGNAL(clicked()), this, SLOT(build()));
}


GLSLMgr::~GLSLMgr() {
    QSettings settings;
    settings.setValue("showWarnings", m_showWarnings->isChecked());
}


bool GLSLMgr::initialize() {
    build();
    if (!m_buildStatus || ((m_buildStatus == 1) && m_showWarnings->isChecked())) {
        exec();
    }
    return m_buildStatus > 0;
}


void GLSLMgr::replaceInSource(const QString& pattern, const QString& text) {
	QRegExp rx(pattern); 
	for (int i = 0; i < m_srcText.size(); ++i) {
		QString name = m_srcText[i]->documentTitle();
		QString src = m_srcText[i]->toPlainText();
		src.replace(rx, text);
		m_srcText[i]->setPlainText(src);
		m_srcText[i]->setDocumentTitle(name);
	}
	build();
}


void GLSLMgr::build() {
    parent()->makeCurrent();

    m_buildStatus = 2;
    m_log = "";
    m_log += QString("GL_VENDOR:                    %1\n").arg((const char*)glGetString(GL_VENDOR));
    m_log += QString("GL_RENDERER:                  %1\n").arg((const char*)glGetString(GL_RENDERER));
    m_log += QString("GL_VERSION:                   %1\n").arg((const char*)glGetString(GL_VERSION));
    m_log += QString("GL_SHADING_LANGUAGE_VERSION:  %1\n\n").arg((const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    for (int i = 0; i < m_srcText.size(); ++i) {
        QString name = m_srcText[i]->documentTitle();
        QByteArray src = m_srcText[i]->toPlainText().toLatin1();

        m_log += QString("---- %1 ----\n").arg(name);
		{
			GLuint shader_id = glCreateShader(GL_FRAGMENT_SHADER);
			const char* src_ptr = src.data();
			const GLint src_len = src.length();
            glShaderSource(shader_id, 1, &src_ptr, &src_len);
			glCompileShader(shader_id);

			GLint len;
			glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &len);
			if (len > 1) {
				char *buf = new char[len];
				glGetShaderInfoLog(shader_id, len, NULL, buf);
                m_log += QString("Compiling:\n");
                m_log += QString("%1\n").arg(buf);
				delete buf;
			}

			GLint compile_status;
			glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compile_status);
            if (!compile_status)
                m_buildStatus = 0;
			
            if (compile_status) {
				GLuint prog_id = glCreateProgram();
				m_pid[name] = prog_id;
				glAttachShader(prog_id, shader_id);
				glLinkProgram(prog_id);

				GLint len = 0;
				glGetProgramiv(prog_id, GL_INFO_LOG_LENGTH, &len);
				if (len > 1) {
					char *buf = new char[len];
					glGetProgramInfoLog(prog_id, len, NULL, buf);
                    m_log += QString("Linking:\n");
                    m_log += QString("%1\n").arg(buf);
					delete buf;
				}

				GLint link_status;
				glGetProgramiv(prog_id, GL_LINK_STATUS, &link_status);
				if (!link_status) m_buildStatus = 0;
			}
		}
        m_log += "\n";
	}

    if (m_buildStatus && m_log.contains("warning", Qt::CaseInsensitive)) {
        m_buildStatus = 1;
    }
    m_icon->setPixmap(QMessageBox::standardIcon(
        m_buildStatus==0? QMessageBox::Critical : 
        m_buildStatus==1? QMessageBox::Warning : 
                          QMessageBox::Information));
    m_logText->setPlainText(m_log);
    m_build->setEnabled(false);
    m_tab->setCurrentIndex(0);
    for (int i = 0; i < m_srcText.size(); ++i) {
        m_srcText[i]->document()->setModified(false);
    }
}


void GLSLMgr::enableEdit() {
    if (m_tab->count() == 1) {
        for (int i = 0; i < m_srcText.size(); ++i) {
            m_tab->addTab(m_srcText[i], m_srcText[i]->documentTitle());
        }
        m_build->show();
    }
}


void GLSLMgr::sourceChanged(bool change) {
    if (change) {
        m_build->setEnabled(true);
    }
}


void GLSLMgr::copyToClipboard() {
    if (m_tab->currentIndex() == 0) {
        QApplication::clipboard()->setText(m_logText->toPlainText());
    } else {
        QTextEdit *text = (QTextEdit*)m_tab->currentWidget();
        QApplication::clipboard()->setText(text->toPlainText());
    }
}
