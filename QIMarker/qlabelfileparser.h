#ifndef QLABELFILEPARSER_H
#define QLABELFILEPARSER_H

#include "qclasslabel.h"
#include <QList>
#include <QString>

class QLabelFileParser
{
public:
    explicit QLabelFileParser(QString _filename);

    QList<QClassLabel> getClassesList();

    static QList<QClassLabel> parseFile(QString _filename);

private:
    QString m_filename;
};

#endif // QLABELFILEPARSER_H
