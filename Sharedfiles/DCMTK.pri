DCMTK_PATH = "C:/Program Files (x86)/DCMTK"

INCLUDEPATH += $${DCMTK_PATH}/include

LIBS += -L$${DCMTK_PATH}/bin \
        -L$${DCMTK_PATH}/lib

LIBS += -ldcmtk
