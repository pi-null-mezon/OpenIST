CONFIG += enableopenmp
enableopenmp {
    win32-msvc* {
        QMAKE_CXXFLAGS+= -openmp
message(OpenMP enabled)
    }
    win32-g++ {
        QMAKE_CXXFLAGS+= -fopenmp
        LIBS += -fopenmp
message(OpenMP enabled)
    }
    linux-g++ {
        QMAKE_CXXFLAGS+= -fopenmp
        LIBS += -fopenmp
message(OpenMP enabled)
    }
}

