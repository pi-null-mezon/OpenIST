TINY_DNN_VERSION = 1.0.0

win32 {
    TINY_DNN_PATH = C:/Programming/3rdParties/tiny-dnn-$${TINY_DNN_VERSION}
    INCLUDEPATH += $${TINY_DNN_PATH}

    # This flag is needed when serialization is enabled in tiny_cnn\config.h
    #QMAKE_CXXFLAGS += -bobjct
}
