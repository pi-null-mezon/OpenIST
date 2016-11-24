TINY_DNN_VERSION = 1.0.0

#Comment to use float precision for tiny_dnn::network weights
DEFINES += CNN_USE_DOUBLE

win32 {
    TINY_DNN_PATH = C:/Programming/3rdParties/tiny-dnn-$${TINY_DNN_VERSION}

    HEADERS += $${TINY_DNN_PATH}/tiny_dnn/config.h
    INCLUDEPATH += $${TINY_DNN_PATH}
}
