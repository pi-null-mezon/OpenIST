#ifndef QIMAGEDIRECTORY_H
#define QIMAGEDIRECTORY_H

#include <QObject>
#include <QDir>
#include <QUrl>
#include <QVariant>
#include <QStringList>

#include "qclasslabel.h"

#define NAME_OF_SUBDIR_FOR_MARKED_IMAGES "Marked"

class QImageDirectory: public QObject
{
    Q_OBJECT
public:
    explicit QImageDirectory(QObject *parent=0);

public slots:
    void setDirUrl(QUrl _url);
    void getNextFile();
    void saveFile();
    void setClassList(QList<QObject *> *_labelslist);

signals:
    void fileUrl(QVariant _url);


private:
    void __clearFilesData();
    void __clearLabels();
    void __saveLabelsInfoFile();

    QDir m_dir;
    QStringList v_files;
    int m_filepos;
    QList<QObject*> *p_labelslist = NULL;
    QString m_filename;
};

#endif // QIMAGEDIRECTORY_H
