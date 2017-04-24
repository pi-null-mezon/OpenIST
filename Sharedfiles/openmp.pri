CONFIG += enableopenmp
enableopenmp {
    win32-msvc* {
        QMAKE_CXXFLAGS += -openmp
    }
    win32-g++ {
        QMAKE_CXXFLAGS += -fopenmp
        LIBS += -fopenmp
    }
    linux {
        QMAKE_CXXFLAGS += -fopenmp
        QMAKE_LFLAGS   += -fopenmp
        LIBS += -lgomp -lpthread
    }
message(OpenMP enabled)
}

