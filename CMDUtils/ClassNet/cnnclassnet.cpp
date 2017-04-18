#include "cnnclassnet.h"
#include <opencv2/highgui.hpp>

using namespace activation;
//-------------------------------------------------------------------------------------------------------
CNNClassificator::CNNClassificator()
{}
//-------------------------------------------------------------------------------------------------------
CNNClassificator::~CNNClassificator()
{}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
network<sequential> CNNClassificator::__createNet(const cv::Size &size, int inchannels, int outchannels)
{  
    int _kernels = 16;
    network<sequential> _net;
    _net << convolutional_layer<>(size.width, size.height, 3, inchannels, _kernels, padding::same)
         << convolutional_layer<relu>(size.width, size.height, 3, _kernels, _kernels, padding::same)
         << max_pooling_layer<identity>(size.width, size.height, _kernels, 2)
         << convolutional_layer<>(size.width/2, size.height/2, 3, _kernels, 2*_kernels, padding::same)
         << convolutional_layer<relu>(size.width/2, size.height/2, 3, 2*_kernels, 2*_kernels, padding::same)
         << max_pooling_layer<identity>(size.width/2, size.height/2, 2*_kernels, 2)
         << fully_connected_layer<relu>(size.width/4 * size.height/4 * 2 * _kernels, 4*_kernels)
         << dropout_layer(4*_kernels, 0.5)
         << fully_connected_layer<softmax>(4*_kernels, outchannels);
    return _net;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNClassificator::train(cv::InputArrayOfArrays _vraw, const std::vector<label_t> &_vlabel, int _epoch, int _minibatch, int _t2cprop)
{
    __train(_vraw, _vlabel, _epoch, _minibatch, _t2cprop, false);
}
void CNNClassificator::update(cv::InputArrayOfArrays _vraw, const std::vector<label_t> &_vlabel, int _epoch, int _minibatch, int _t2cprop)
{
    __train(_vraw, _vlabel, _epoch, _minibatch, _t2cprop, true);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNClassificator::__train(cv::InputArrayOfArrays _vraw, const std::vector<label_t> &_vlabel, int _epoch, int _minibatch, int _t2cprop, bool preservedata)
{
    if(_vraw.kind() != cv::_InputArray::STD_VECTOR_MAT) {
        cv::String error_message = "CNNClassificator warning! The images are expected as InputArray::STD_VECTOR_MAT!";
        CV_Error(cv::Error::StsBadArg, error_message);
    }
    if(_vraw.total() == 0 || _vlabel.size() == 0) {
        cv::String error_message = cv::format("CNNClassificator warning! Empty training data was given!");
        CV_Error(cv::Error::StsUnsupportedFormat, error_message);
    }

    // Get the vector of visual images
    std::vector<cv::Mat> srcmats;
    std::vector<tiny_dnn::vec_t> srcvec_t;
    _vraw.getMatVector(srcmats);
    for(size_t it = 0; it < srcmats.size(); it++) {
        if(it == 0) {
            if(m_inputchannels == 0)
                setInputChannels( srcmats[it].channels() );
            if(m_inputsize.area() == 0)
                setInputSize(cv::Size((srcmats[it]).cols,(srcmats[it]).rows));
        }
        srcvec_t.push_back( __mat2vec_t(srcmats[it], m_inputsize, m_irm, m_inputchannels) );
    }

    // Get the vector of labels
    std::vector<label_t> srclabels(_vlabel.begin(), _vlabel.end());

    // Shuffle and unskew input pairs, it should prevent one class overfitting
    std::vector<label_t> ulbls, tlbls, clbls;
    std::vector<tiny_dnn::vec_t> uvects, tvects, cvects;
    size_t _uniquelables = 0;
    __shuffle_and_unskew(srcvec_t, srclabels, uvects, ulbls, &_uniquelables);
    // The number of the output channels should be equal (or greater) than the number of unique labels
    setOutputChannels(static_cast<int>(_uniquelables));
    // Clear memory
    srcvec_t.clear();
    srclabels.clear();
    // Divide samples into training and control sets
    __subsetdata(ulbls, _t2cprop, tlbls, clbls);
    __subsetdata(uvects, _t2cprop, tvects, cvects);
    // Clear memory
    ulbls.clear();
    uvects.clear();

    std::cout << std::endl << "Metadata:" << std::endl
              << " - inchannels " << getInputChannels() << std::endl
              << " - unique labels in dataset " << _uniquelables << std::endl
              << " - number of samples selected for training " << tvects.size() << std::endl
              << " - number of samples selected for control " << cvects.size()  << std::endl;

    if(preservedata == false) {
        m_net = __createNet(m_inputsize, m_inputchannels, m_outputchannels);
    } else {
        std::cout << "Checking performance of the network before update:" << std::endl;
        // This was added to control network weights before updating
        if(clbls.size() > 0) {
            tiny_dnn::result cresult = m_net.test(cvects,clbls);
            tiny_dnn::result tresult = m_net.test(tvects,tlbls);
            std::cout << " - accuracy on training set: " << tresult.accuracy() << std::endl
                      << " - accuracy on control set: "  << cresult.accuracy() << std::endl;
        }
    }
    
    adam _opt;
    // Note that for right learning by the fit() function, the values in label data (segvec_t here) should be normalized to [0.0; 1.0] interval
    std::cout << std::endl << " Epoch\tAccuracy (training / control)" << std::endl;
    m_net.train<cross_entropy>(_opt, tvects, tlbls, static_cast<size_t>(_minibatch), _epoch,
                                    [](){},
                                    [&](){
                                            static int epoch = 0;
                                            if(((epoch % 1) == 0) || (epoch == _epoch - 1)) {
                                                if(clbls.size() > 0) {
                                                    tiny_dnn::result cresult = m_net.test(cvects,clbls);
                                                    tiny_dnn::result tresult = m_net.test(tvects,tlbls);
                                                    std::cout << "  " << epoch << "\t" << tresult.accuracy() << " / " << cresult.accuracy() << std::endl;
                                                } else {
                                                    tiny_dnn::result tresult = m_net.test(tvects,tlbls);
                                                    std::cout << " " << epoch << "\t" << tresult.accuracy() << " / unknown" << std::endl;
                                                }
                                            }
                                            epoch++;
                                          });

}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNClassificator::setInputChannels(int _value)
{
    m_inputchannels = _value;
}
void CNNClassificator::setOutputChannels(int _value)
{
    m_outputchannels = _value;
}
int CNNClassificator::getInputChannels() const
{
    return m_inputchannels;
}
int CNNClassificator::getOutputChannels() const
{
    return m_outputchannels;
}
void CNNClassificator::setInputSize(const cv::Size &_size)
{
    m_inputsize = _size;
}
cv::Size CNNClassificator::getInputSize() const
{
    return m_inputsize;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNClassificator::save(const char *filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if(fs.isOpened()) {

        fs << "typename" << m_uniquename;
        fs << "width" << m_inputsize.width;
        fs << "height" << m_inputsize.height;
        fs << "inchannels" << m_inputchannels;
        fs << "outchannels" << m_outputchannels;

        std::vector<tiny_dnn::float_t> _weights;
        std::vector<tiny_dnn::vec_t*> _w;
        for(size_t i = 0; i < m_net.depth(); i++) {
            _w = m_net[i]->weights();
            tiny_dnn::vec_t *_v;
            for(size_t j = 0; j < _w.size(); j++) {
                _v = _w[j];
                _weights.insert(_weights.end(), _v->begin(), _v->end());
            }
        }
        fs << "weights" << _weights;

        fs.release();
    }
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
bool CNNClassificator::load(const char *filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(fs.isOpened()) {

        cv::String _name;
        fs["typename"] >> _name;
        if(_name != m_uniquename) {
            cv::String error_message = cv::format("CNNClassificator! - > Can not load %s data into %s.", _name.c_str(), m_uniquename.c_str());
            CV_Error(cv::Error::StsBadArg, error_message);
            fs.release();
            return false;
        }

        int w, h;
        fs["width"] >> w;
        fs["height"] >> h;
        setInputSize(cv::Size(w,h));
        fs["inchannels"] >> m_inputchannels;
        fs["outchannels"] >> m_outputchannels;

        m_net = __createNet(m_inputsize, m_inputchannels, m_outputchannels);        

        std::vector<tiny_dnn::float_t> _weights;
        fs["weights"] >> _weights;
        int idx = 0;
        std::vector<tiny_dnn::vec_t*> _w;
        for(size_t i = 0; i < m_net.depth(); i++) {
            _w = m_net[i]->weights();
            tiny_dnn::vec_t *_v;
            for(size_t j = 0; j < _w.size(); j++) {
                _v = _w[j];
                for(size_t k = 0; k < _v->size(); k++)
                    _v->at(k) = _weights[idx++];               
            }
        }


        fs.release();
        return true;
    }
    return false;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
label_t CNNClassificator::predict(const cv::Mat &image) const
{
    m_net.set_netphase(tiny_dnn::net_phase::test);
    vec_t res = m_net.predict( __mat2vec_t(image, m_inputsize, m_irm, m_inputchannels) );
    for(size_t i = 0; i < res.size(); i++)
        std::cout << "Probability for calss " << i << " is: " << res[i] << std::endl;
    return m_net.predict_label( __mat2vec_t(image, m_inputsize, m_irm, m_inputchannels) );
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
tiny_dnn::vec_t __mat2vec_t(const cv::Mat &img, const cv::Size targetSize, ImageResizeMethod resizeMethod, int targetChannels, double min, double max)
{
    cv::Mat _mat;
    // Equalize histogram to normalize brightness distribution across all samples
    if(img.channels() == 1) {
        cv::equalizeHist(img,_mat);
    } else {
        _mat = img;
    }

    // Change channels quantity if needed
    if((targetChannels > 0) && (targetChannels != _mat.channels()))
        switch(targetChannels) {
            case 1:
                if(_mat.channels() == 3)
                    cv::cvtColor(_mat, _mat, CV_BGR2GRAY);
                else if(_mat.channels() == 4)
                    cv::cvtColor(_mat, _mat, CV_BGRA2GRAY);
                // Equalize histogram to normalize brightness distribution across all samples
                cv::equalizeHist(_mat,_mat);
                break;
            case 3:
                cv::cvtColor(_mat, _mat, CV_GRAY2BGR);
                break;
        }

    // Resize if needed
    if((targetSize.area() > 0)  && (img.cols != targetSize.width || img.rows != targetSize.height)) {
        switch(resizeMethod){
            case ImageResizeMethod::CropAndResizeFromCenter:
                _mat =__cropresize(_mat, targetSize);
                break;
            case ImageResizeMethod::PaddZeroAndResize:
                _mat = __propresize(_mat, targetSize);
                break;
        }
    }

    // Visualize
    cv::namedWindow("CNNClassificator", CV_WINDOW_NORMAL);
    cv::imshow("CNNClassificator", _mat);
    cv::waitKey(1);

    // Convert to float_t type    
    int _maxval = 1;
    switch(_mat.depth()) {
        case CV_8U:
            _maxval = 255;
            break;
        case CV_16U:
            _maxval = 65535;
            break;
    }
    _mat.convertTo(_mat, (sizeof(tiny_dnn::float_t) == sizeof(double)) ? CV_64F : CV_32F, (max-min)/_maxval, min);


    // Construct vec_t image representation
    tiny_dnn::vec_t ovect;
    switch(_mat.channels()) {
        case 1: {
            tiny_dnn::float_t *ptr = _mat.ptr<tiny_dnn::float_t>(0);
            ovect = tiny_dnn::vec_t(ptr, ptr + _mat.cols * _mat.rows );
        } break;
        case 3: {
            std::vector<cv::Mat> _vmats;
            cv::split(_mat, _vmats);
            for(int i = 0; i < 3; i++) {
                cv::Mat _chanmat = _vmats[i];
                tiny_dnn::float_t *ptr = _chanmat.ptr<tiny_dnn::float_t>(0);
                ovect.insert(ovect.end(), ptr, ptr + _mat.cols * _mat.rows);
            }
        } break;
    }

    return ovect;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
cv::Mat __cropresize(const cv::Mat &input, const cv::Size size)
{
    cv::Mat output;
    if(size.area() > 0){
        cv::Rect2f roiRect(0,0,0,0);
        if( (float)input.cols/input.rows > (float)size.width/size.height) {
            roiRect.height = (float)input.rows;
            roiRect.width = input.rows * (float)size.width/size.height;
            roiRect.x = (input.cols - roiRect.width)/2.0f;
        } else {
            roiRect.width = (float)input.cols;
            roiRect.height = input.cols * (float)size.height/size.width;
            roiRect.y = (input.rows - roiRect.height)/2.0f;
        }
        roiRect &= cv::Rect2f(0,0,(float)input.cols,(float)input.rows);
        if(roiRect.area() > 0)  {
            cv::Mat croppedImg(input, roiRect);
            int interpolationMethod = 0;
            if(size.area() > roiRect.area())
                interpolationMethod = CV_INTER_LANCZOS4;
            else
                interpolationMethod = CV_INTER_AREA;
            cv::resize(croppedImg, output, size, 0, 0, interpolationMethod);
        }
    } else {
        output = input;
    }
    return output;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
cv::Mat __propresize(const cv::Mat &input, const cv::Size size)
{
    cv::Mat output;
    if(size.area() > 0){
        cv::Rect2f roiRect(0,0,0,0);
        cv::Point shift(0,0);
        if( (float)input.cols/input.rows > (float)size.width/size.height) {
            roiRect.width = (float)input.cols;
            roiRect.height = input.cols * (float)size.height/size.width;
            shift.y = static_cast<int>((roiRect.height - (float)input.rows)/2.0f);
        } else {
            roiRect.height = (float)input.rows;
            roiRect.width = input.rows * (float)size.width/size.height;
            shift.x = static_cast<int>((roiRect.width - (float)input.cols)/2.0f);
        }
        output = cv::Mat::zeros(roiRect.size(),input.type());
        cv::Mat imgcontent = output(cv::Rect(0,0,input.cols,input.rows)+shift);
        input.copyTo(imgcontent);
        if(roiRect.area() > 0)  {
            int interpolationMethod = 0;
            if(size.area() > roiRect.area())
                interpolationMethod = CV_INTER_LANCZOS4;
            else
                interpolationMethod = CV_INTER_AREA;
            cv::resize(output, output, size, 0, 0, interpolationMethod);
        }
    } else {
        output = input;
    }
    return output;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
cv::Mat __restoreSize (const cv::Mat &img, const cv::Size &dstSize, ImageResizeMethod resizeMethod)
{
    if(resizeMethod == ImageResizeMethod::PaddZeroAndResize)
        return __cropresize(img, dstSize);
    else
        return __propresize(img, dstSize);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNClassificator::setImageResizeMethod(ImageResizeMethod method)
{
    m_irm = method;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
ImageResizeMethod CNNClassificator::getImageResizeMethod() const
{
    return m_irm;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNClassificator::setUniqueName(const cv::String &_name)
{
    m_uniquename = _name;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void visualizeActivations(const tiny_dnn::network<tiny_dnn::sequential> &_net)
{
    for(size_t i = 0; i <_net.depth(); i++) {
        tiny_dnn::image<unsigned char> img = _net[i]->output_to_image(); // visualize activations of recent input
        cv::Mat mat = tinyimage2mat(img);
        if(mat.empty() == false) {
            cv::String windowname = (std::string("Activation of the layer ") + std::to_string(i)).c_str();
            cv::namedWindow(windowname, CV_WINDOW_NORMAL);
            cv::imshow(windowname, mat);
        }
    }
    cv::waitKey(1);

}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void visualizeLastLayerActivation(const tiny_dnn::network<tiny_dnn::sequential> &_net)
{
    tiny_dnn::image<unsigned char> img = _net[_net.depth()-1]->output_to_image(); // visualize activations of recent input
    cv::Mat mat = tinyimage2mat(img);
    if(mat.empty() == false) {
        cv::namedWindow("Last layer activation", CV_WINDOW_NORMAL);
        cv::imshow("Last layer activation", mat);
    }
    cv::waitKey(1);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
template<typename T>
cv::Mat tinyimage2mat(const tiny_dnn::image<T> &_image)
{
    std::vector<T> data = _image.data();
    cv::Mat mat = cv::Mat(static_cast<int>(_image.height()), static_cast<int>(_image.width()), CV_8UC1, &data[0]);
    return mat.clone();
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
template <typename Iterator1, typename Iterator2>
void __random_shuffle (Iterator1 v1first, Iterator1 v1last, Iterator2 v2first, Iterator2 v2last)
{
    std::iterator_traits<Iterator1>::difference_type i, v1length = v1last - v1first;
    std::iterator_traits<Iterator2>::difference_type v2length = v2last - v2first;
    if(v1length != v2length) {
        CV_Error(cv::Error::StsBadArg, "Error in __random_shuffle(): input vectors have different sizes");
        return;
    } else {
        int pos;
        for(i = 0; i < v1length; i++) {
            pos = std::rand() % v1length;
            std::swap(v1first[i],v1first[pos]);
            std::swap(v2first[i],v2first[pos]);
        }
    }
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
template<typename T1, typename T2>
void __shuffle_and_unskew(const std::vector<T1> &vraw, const std::vector<T2> &vlabel, std::vector<T1> &_outraw, std::vector<T2> &_outlabel, size_t *_ulabels)
{
    if(vraw.size() != vlabel.size()) {
        CV_Error(cv::Error::StsBadArg, "Error in __unskew(): input vectors have different sizes!");
        return;
    }
    // Determine number of the unique labels in the input data
    std::vector<T2> _templabel = vlabel;
    std::sort(_templabel.begin(), _templabel.end());
    auto _last = std::unique(_templabel.begin(), _templabel.end());
    size_t uniquelabels = static_cast<size_t>(_last - _templabel.begin());
    if(_ulabels)
        *_ulabels = uniquelabels;

    // Create vector of empty vectors with size equal to the number of unique labels
    // vectors of this vector will store positions of the images for appropriate label
    std::vector<std::vector<size_t>> v_imagesforlabelpos(uniquelabels, std::vector<size_t>());

    // Fill vectors of positions, labels should form continuous row, i.e. {0, 1, 2, 3,...}
    for(size_t i = 0; i < vlabel.size(); i++)
        (v_imagesforlabelpos[ vlabel[i] ]).push_back(i);

    // Determine label with minimum samples quantity size, then we will use this minimum size for output construction
    std::vector<size_t> v_vectorsizes(uniquelabels, 0);
    for(size_t i = 0; i < v_imagesforlabelpos.size(); i++)
        v_vectorsizes[i] = (v_imagesforlabelpos[i]).size();
    size_t minsize = *std::min_element(v_vectorsizes.begin(), v_vectorsizes.end());

    // Copy data to the output vectors
    for(size_t n = 0; n < minsize; n++) {
        for(size_t i = 0; i < v_imagesforlabelpos.size(); i++) {
            _outlabel.push_back(i);
            _outraw.push_back( vraw[ v_imagesforlabelpos[i][n] ] );
        }
    }
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
template<typename T>
void __subsetdata(const std::vector<T> &_vin, int _mod, std::vector<T> &_vbig, std::vector<T> &_vsmall)
{
    if(_mod != 0) {
        for(size_t i = 0; i < _vin.size(); i++) {
            if((i % _mod) == 0)
                _vsmall.push_back(_vin[i]);
            else
                _vbig.push_back(_vin[i]);
        }
    } else {
        _vbig = std::vector<T>(_vin.begin(), _vin.end());
    }
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------



