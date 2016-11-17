#include "qimagedirectory.h"
#include <QTextStream>
#include <QDateTime>

QImageDirectory::QImageDirectory(QObject *parent) : QObject(parent)
{
    __clearFilesData();
}

void QImageDirectory::setDirUrl(QUrl _url)
{
    __clearFilesData();

    m_dir = QDir(_url.toLocalFile());

    if(m_dir.exists() == false) {

        qWarning("QImageDirectory: Directory does not exist!");
        return;

    } else {

        QStringList _filters;
        _filters << "*.jpg" << "*.png" << "*.bmp";
        m_dir.setNameFilters(_filters);
        v_files = m_dir.entryList(_filters, QDir::Files | QDir::NoDotAndDotDot);

        if(v_files.size() > 0) {

            m_dir.mkdir(NAME_OF_SUBDIR_FOR_MARKED_IMAGES);

            qInfo("QImageDirectory: %d files in directory", v_files.size());
            __saveLabelsInfoFile();
            getNextFile();

        } else {
            qInfo("QImageDirectory: There is no images in the directory!");
        }

    }
}

void QImageDirectory::getNextFile()
{
    if(v_files.size() > 0 && m_filepos < v_files.size()) {
        __clearLabels();
        m_filename = m_dir.absoluteFilePath( v_files[m_filepos++] );
        emit fileUrl( QVariant::fromValue(QUrl::fromLocalFile(m_filename)) );
    } else {
        qInfo("No more files in directory");
        m_filename = QString();
        emit fileUrl(QVariant::fromValue(QString("qrc:/Resources/Images/Startup.png")));
    }
}

void QImageDirectory::saveFile()
{
    if(p_labelslist) {

        if(!m_filename.isEmpty() && m_filepos > 0 && m_filepos < v_files.size() ) {

            QString _extension = (v_files[m_filepos-1]).section('.',1,1);
            QString _filename = (v_files[m_filepos-1]).section('.',0,0);
            _filename.append("(");

            bool _firstlabel = true;
            for(int i = 0; i < p_labelslist->size(); i++) {
                QClassLabel *_plabel =  qobject_cast<QClassLabel *>((p_labelslist->at(i)));
                if(_plabel->checked())
                    if(_firstlabel) {
                        _filename.append(QString("%1").arg(QString::number(_plabel->id())));
                        _firstlabel = false;
                    }
                    else
                        _filename.append(QString("_%1").arg(QString::number(_plabel->id())));
            }

            _filename.append(QString(").") + _extension);
            _filename.prepend( m_dir.absolutePath() + QString("/") + QString(NAME_OF_SUBDIR_FOR_MARKED_IMAGES) + QString("/"));
            QFile _imgFile(m_filename);

            if(_imgFile.copy(_filename) == false)
                qWarning("Can not save %s", _filename.toLocal8Bit().constData());
        } else {
            qWarning("Nothing to save!");
        }

    } else {
        qWarning("Empty labels list!");
    }

}

void QImageDirectory::setClassList(QList<QObject *> *_labelslist)
{
    if(_labelslist) {
        p_labelslist = _labelslist;
        qInfo("Labels list contains %d", p_labelslist->size());
    }
    else
        qWarning("Null labels list!");
}

void QImageDirectory::__clearFilesData()
{
    m_filepos = 0;
    v_files = QStringList();
    m_filename = QString();
}

void QImageDirectory::__clearLabels()
{
    if(p_labelslist) {
        for(int i = 0; i < p_labelslist->size(); i++) {
            QClassLabel *_plabel =  qobject_cast<QClassLabel *>((p_labelslist->at(i)));
            if(_plabel->checked())
                _plabel->setchecked(false);
        }
    }
}

void QImageDirectory::__saveLabelsInfoFile()
{
    if(p_labelslist) {

        if(p_labelslist->size() > 0) {
            QFile _infofile(m_dir.absolutePath() + QString("/") + QString(NAME_OF_SUBDIR_FOR_MARKED_IMAGES) + QString("/Labels.txt"));
            if(_infofile.open(QFile::WriteOnly) == false) {
                qWarning("QImageDirectory: Can not save info file in %s!", _infofile.fileName().toLocal8Bit().constData());
                return;
            }

            QTextStream _stream(&_infofile);
            _stream << "# Labels info file, created by QIMarker at" << QDateTime::currentDateTime().toString("dd.MM.yyyy hh:mm:ss")
                    << "\n\n#\tID\t\t\tName\t\t\tDescription";
            for(int i = 0; i < p_labelslist->size(); i++) {
                QClassLabel *_plabel =  qobject_cast<QClassLabel *>((p_labelslist->at(i)));
                _stream << "\n\n\t" << _plabel->id() << "\t\t\t" << _plabel->name() << "\t\t\t" << _plabel->description();
            }

            _stream.flush();
            _infofile.close();
        }

    }
}

