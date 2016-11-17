#include "qlabelfileparser.h"
#include <QFile>

QLabelFileParser::QLabelFileParser(QString _filename)
{
    m_filename = _filename;
}

QList<QClassLabel> QLabelFileParser::getClassesList()
{
    return parseFile(m_filename);
}

QList<QClassLabel> QLabelFileParser::parseFile(QString _filename)
{
    QFile _file(_filename);

    if(_file.open(QFile::ReadOnly) == false) {
        qWarning("QLabelFileParser: can not open file!");
        return QList<QClassLabel>();
    }

    QList<QClassLabel> _list;

    while(true) {
        QByteArray _line = _file.readLine();
        if(_line.size() == 0) {
            qInfo("End of file has been reached");
            break;
        } else {

            if(_line[0] == '#') {
                // Do nothing it is a comment
            } else if (_line.contains('"')) {
                QString _string(_line);

                int _id = _string.section('"',0,0).toInt();
                QString _name = _string.section('"',1,1).section('"',0,0);
                QString _description = _string.section('"',3,3).section('"',0,0);

                qInfo("QLabelFileParser: new class has been found >> id: %d; name: %s; description: %s",
                      _id,_name.toLocal8Bit().constData(), _description.toLocal8Bit().constData());

                _list.push_back( QClassLabel (_id,_name,_description) );
            }

        }
    }

    _file.close();

    return _list;
}


