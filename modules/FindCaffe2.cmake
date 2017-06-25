unset(Caffe2_FOUND)

find_path(Caffe2_INCLUDE_DIR NAMES 
caffe2/core/blob.h
caffe2/core/net.h
caffe2/core/tensor.h
caffe2/core/operator.h
caffe2/core/predictor.h
caffe2/core/workspace.h
HINTS
/usr/local/include)

find_library(Caffe2_LIBS NAMES 
Caffe2_CPU
Caffe2_GPU
HINTS
/usr/local/lib)

if(Caffe2_LIBS AND Caffe2_INCLUDE_DIR)
    set(Caffe2_FOUND 1)
endif()

