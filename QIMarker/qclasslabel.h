#ifndef QCLASSLABEL_H
#define QCLASSLABEL_H

#include <QObject>

class QClassLabel : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QString name READ name WRITE setname NOTIFY nameChanged)
    Q_PROPERTY(bool checked READ checked WRITE setchecked NOTIFY checkedChanged)
    Q_PROPERTY(QString description READ description WRITE setdescription NOTIFY descriptionChanged)
    Q_PROPERTY(int id READ id WRITE setid NOTIFY idChanged)

public:
    explicit QClassLabel(int _id=-1, QString _name=QString(), QString _description=QString(), QObject *parent = 0);
    explicit QClassLabel(const QClassLabel &_classlabel);
    QString name() const;
    bool checked() const;
    QString description() const;    
    int id() const;

signals:
    void nameChanged(QString _name);
    void checkedChanged(bool _checked);
    void descriptionChanged(QString _description);
    void idChanged(int id);

public slots:
    void setname(QString _name);
    void setchecked(bool _checked);
    void setdescription(QString _description);
    void setid(int id);

private:
    QString m_name;
    bool m_checked;
    QString m_description;
    int m_id;
};

#endif // QCLASSLABEL_H
