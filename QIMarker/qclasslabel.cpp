#include "qclasslabel.h"

QClassLabel::QClassLabel(int _id, QString _name, QString _description, QObject *parent) : QObject(parent),
    m_name(_name),
    m_description(_description),
    m_checked(false),
    m_id(_id)
{
}

QClassLabel::QClassLabel(const QClassLabel &_classlabel)
{
    setid(_classlabel.id());
    setname(_classlabel.name());
    setdescription(_classlabel.description());
    setchecked(_classlabel.checked());
}

QString QClassLabel::name() const
{
    return m_name;
}

bool QClassLabel::checked() const
{
    return m_checked;
}

QString QClassLabel::description() const
{
    return m_description;
}

int QClassLabel::id() const
{
    return m_id;
}

void QClassLabel::setname(QString _name)
{
    if (m_name == _name)
        return;

    m_name = _name;
    emit nameChanged(_name);
}

void QClassLabel::setchecked(bool _checked)
{
    if (m_checked == _checked)
        return;

    m_checked = _checked;
    emit checkedChanged(_checked);
}

void QClassLabel::setdescription(QString _description)
{
    if (m_description == _description)
        return;

    m_description = _description;
    emit descriptionChanged(_description);
}

void QClassLabel::setid(int id)
{
    if (m_id == id)
        return;

    m_id = id;
    emit idChanged(id);
}
