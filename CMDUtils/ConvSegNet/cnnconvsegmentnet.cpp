#include "cnnconvsegmentnet.h"
#include <opencv2/highgui.hpp>

#include <QDebug>

namespace segnet {

using namespace activation;
//-------------------------------------------------------------------------------------------------------
CNNConvSegmentnet::CNNConvSegmentnet()
{}
//-------------------------------------------------------------------------------------------------------
CNNConvSegmentnet::~CNNConvSegmentnet()
{}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
network<sequential> CNNConvSegmentnet::__createNet(const cv::Size &size, int inchannels, int outchannels)
{  
    Q_UNUSED(size);
    Q_UNUSED(inchannels);
    Q_UNUSED(outchannels);

    network<sequential> _net;
    return _net;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::initNet(const cv::Size &size, const int inchannels, const int outchannels, const tiny_cnn::float_t *_weights)
{
    setInputSize(size);
    setInputChannels(inchannels);
    setOutputChannels(outchannels);
    // Create empty network with the appropriate structure
    m_net = __createNet(size, inchannels, outchannels);
    // Load weights into the network
    int idx = 0;
    std::vector<tiny_cnn::vec_t*> _w;
    for(size_t i = 0; i < m_net.depth(); i++) {
        _w = m_net[i]->get_weights();
        tiny_cnn::vec_t *_v;
        for(size_t j = 0; j < _w.size(); j++) {
            _v = _w[j];
            for(size_t k = 0; k < _v->size(); k++)
                _v->at(k) = _weights[idx++];
        }
    }
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::train(cv::InputArrayOfArrays _vraw, cv::InputArrayOfArrays _vlabel, int _epoch, int _minibatch)
{
    __train(_vraw, _vlabel, _epoch, _minibatch, false);
}
void CNNConvSegmentnet::update(cv::InputArrayOfArrays _vraw, cv::InputArrayOfArrays _vlabel, int _epoch, int _minibatch)
{
    __train(_vraw, _vlabel, _epoch, _minibatch, true);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::__train(cv::InputArrayOfArrays _vraw, cv::InputArrayOfArrays _vlabel, int _epoch, int _minibatch, bool preservedata)
{
    if(_vraw.kind() != cv::_InputArray::STD_VECTOR_MAT && _vlabel.kind() != cv::_InputArray::STD_VECTOR_MAT) {
        cv::String error_message = "CNNConvSegmentnet! - > The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>).";
        CV_Error(cv::Error::StsBadArg, error_message);
    }
    if(_vraw.total() == 0 || _vlabel.total() == 0) {
        cv::String error_message = cv::format("CNNConvSegmentnet! - > Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(cv::Error::StsUnsupportedFormat, error_message);
    }

    std::vector<cv::Mat> _vmatrawimg;
    _vraw.getMatVector(_vmatrawimg);
    std::vector<cv::Mat> _vlabelimg;
    _vlabel.getMatVector(_vlabelimg);

    // Check if data is well-aligned
    if(_vmatrawimg.size() != _vlabelimg.size()) {
        cv::String error_message = cv::format("CNNConvSegmentnet! - > The number of samples (src) must be equal to the number of the labels. Was len(samples)=%d, len(labels)=%d.", _vmatrawimg.size(), _vlabelimg.size());
        CV_Error(cv::Error::StsBadArg, error_message);
    }

    std::vector<vec_t> srcvec_t;
    std::vector<label_t> srclabel_t;

    for(size_t it = 0; it < _vmatrawimg.size(); it++) {
        if(it == 0) {
            if(m_inputchannels == 0)
                setInputChannels( (_vmatrawimg[it]).channels() );
            if(m_outputchannels == 0)
                setOutputChannels(2); // 0 and 1 labels
        }

        __mosaic(m_inputsize, _vmatrawimg[it], _vlabelimg[it], srcvec_t, srclabel_t);
    }

    // Shuffle input pairs in random order, it should prevent overfitting when training samples is taken from different sources
    __random_shuffle(srcvec_t.begin(),srcvec_t.end(),srclabel_t.begin(),srclabel_t.end());

    // Unskew dataset (i.e.: make subsampling with equal number of each class volume)
    std::vector<vec_t> _rawimgs;
    std::vector<label_t> _labels;
    __unskew(srcvec_t,srclabel_t,_rawimgs,_labels);
    // clear unused memory
    srcvec_t.clear();
    srclabel_t.clear();
    // Get test subset
    std::vector<vec_t> _raw2train, _raw2test;
    std::vector<label_t> _lbl2train, _lbl2test;
    int mod = 11;
    __subsetdata(_rawimgs, mod, _raw2train, _raw2test);
    __subsetdata(_labels, mod, _lbl2train, _lbl2test);
    // clear unused memory
    _rawimgs.clear();
    _labels.clear();

    if(preservedata == false)
        m_net = __createNet(m_inputsize, m_inputchannels, m_outputchannels);

    // Batch_size is a number of samples enrolled per parameters update
    adam _opt;
    m_net.train<cross_entropy>(_opt, _raw2train, _lbl2train, static_cast<cnn_size_t>(_minibatch), _epoch,
                                        [](){ /*visualizeActivations(m_net);*/},
                                        [&](){
                                                static int ep = 0;
                                                if(ep % 10 == 0) {
                                                    tiny_cnn::result _result = m_net.test(_raw2test, _lbl2test);
                                                    std::cout << "\nEpoch " << ep << " / " << _epoch << " has been passed (accuracy: " << _result.accuracy() << ")";
                                                } ep++;
                                              });
}
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::__mosaic(const cv::Size &_msize, const cv::Mat &_rawimg, const cv::Mat &_labelimg, std::vector<vec_t> &_vparts, std::vector<label_t> &_vmarks)
{
    if(_rawimg.size() != _labelimg.size()) {
        cv::String error_message = cv::format("CNNConvSegmentnet! - > Raw and Label images should have equal sizes but does not have!");
        CV_Error(cv::Error::StsBadArg, error_message);
    }

    for(int y = 0 ; y < _rawimg.rows - _msize.height; y++) {
        for(int x = 0; x < _rawimg.cols - _msize.width; x++) {

            cv::Rect _roirect = cv::Rect(0,0,_msize.width,_msize.height)+cv::Point(x,y);

            cv::Mat _rawmat = _rawimg( _roirect );
            cv::Mat _labelmat = _labelimg( _roirect );

            _vparts.push_back( __mat2vec_t(_rawmat, m_inputchannels) );
            _vmarks.push_back( __getLabel(_labelmat) );
        }
    }

}
//-------------------------------------------------------------------------------------------------------
label_t CNNConvSegmentnet::__getLabel(const cv::Mat &img)
{
    if(img.at<unsigned char>(img.rows/2, img.cols/2) > 128)
        return label_t(1);
    else
        return label_t(0);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::setInputChannels(int _value)
{
    m_inputchannels = _value;
}
void CNNConvSegmentnet::setOutputChannels(int _value)
{
    m_outputchannels = _value;
}
int CNNConvSegmentnet::getInputChannels() const
{
    return m_inputchannels;
}
int CNNConvSegmentnet::getOutputChannels() const
{
    return m_outputchannels;
}
void CNNConvSegmentnet::setInputSize(const cv::Size &_size)
{
    m_inputsize = _size;
}
cv::Size CNNConvSegmentnet::getInputSize() const
{
    return m_inputsize;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::save(const char *filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if(fs.isOpened()) {

        fs << "width" << m_inputsize.width;
        fs << "height" << m_inputsize.height;
        fs << "inchannels" << m_inputchannels;
        fs << "outchannels" << m_outputchannels;

        std::vector<tiny_cnn::float_t> _weights;
        std::vector<tiny_cnn::vec_t*> _w;
        for(size_t i = 0; i < m_net.depth(); i++) {
            _w = m_net[i]->get_weights();
            tiny_cnn::vec_t *_v;
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
bool CNNConvSegmentnet::load(const char *filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(fs.isOpened()) {

        int w, h;
        fs["width"] >> w;
        fs["height"] >> h;
        setInputSize(cv::Size(w,h));
        fs["inchannels"] >> m_inputchannels;
        fs["outchannels"] >> m_outputchannels;

        m_net = __createNet(m_inputsize, m_inputchannels, m_outputchannels);

        std::vector<tiny_cnn::float_t> _weights;
        fs["weights"] >> _weights;
        int idx = 0;
        std::vector<tiny_cnn::vec_t*> _w;
        for(size_t i = 0; i < m_net.depth(); i++) {
            _w = m_net[i]->get_weights();
            tiny_cnn::vec_t *_v;
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
cv::Mat CNNConvSegmentnet::predict(const cv::Mat &image) const
{
    m_net.set_netphase(tiny_cnn::net_phase::test);

    cv::Mat _output = cv::Mat::zeros(image.rows - m_inputsize.height, image.cols - m_inputsize.width, CV_8UC1);

    cv::Rect _roirect(0,0,m_inputsize.width, m_inputsize.height);
    for(int y = 0; y < image.rows - m_inputsize.height; y++) {
        for(int x = 0; x < image.cols - m_inputsize.width; x++) {
            cv::Mat _tempmat = image(_roirect + cv::Point(x,y));
            vec_t _vp = m_net.predict( __mat2vec_t(_tempmat, m_inputchannels) );
            _output.at<unsigned char>(y,x) = static_cast<unsigned char>(_vp[1]*255);
        }
    }
    return _output;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
tiny_cnn::vec_t __mat2vec_t(const cv::Mat &img, int targetChannels, double min, double max)
{
    // Resize if needed
    cv::Mat _mat = img;

    // Change channels quantity if needed
    if((targetChannels > 0) && (targetChannels != _mat.channels()))
        switch(targetChannels) {
            case 1:
                if(_mat.channels() == 3)
                    cv::cvtColor(_mat, _mat, CV_BGR2GRAY);
                else
                    cv::cvtColor(_mat, _mat, CV_BGRA2GRAY);
                break;
            case 3:
                cv::cvtColor(_mat, _mat, CV_GRAY2BGR);
                break;
        }

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
    _mat.convertTo(_mat, (sizeof(tiny_cnn::float_t) == sizeof(double)) ? CV_64F : CV_32F, (max-min)/_maxval, min);

    // Visualize
    /*cv::namedWindow("CNNConvSegmentnet", CV_WINDOW_NORMAL);
    cv::imshow("CNNConvSegmentnet", _mat);
    cv::waitKey(1);*/

    // Construct vec_t image representation
    tiny_cnn::vec_t ovect;
    switch(_mat.channels()) {
        case 1: {
            tiny_cnn::float_t *ptr = _mat.ptr<tiny_cnn::float_t>(0);
            ovect = tiny_cnn::vec_t(ptr, ptr + _mat.cols * _mat.rows );
        } break;
        case 3: {
            std::vector<cv::Mat> _vmats;
            cv::split(_mat, _vmats);
            for(int i = 0; i < 3; i++) {
                cv::Mat _chanmat = _vmats[i];
                tiny_cnn::float_t *ptr = _chanmat.ptr<tiny_cnn::float_t>(0);
                ovect.insert(ovect.end(), ptr, ptr + _mat.cols * _mat.rows);
            }
        } break;
    }

    return ovect;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void visualizeActivations(const tiny_cnn::network<tiny_cnn::sequential> &_net)
{
    for(size_t i = 0; i <_net.depth(); i++) {
        tiny_cnn::image<unsigned char> img = _net[i]->output_to_image(); // visualize activations of recent input
        cv::Mat mat = tinyimage2mat(img);
        if(mat.empty() == false) {
            cv::String windowname = (std::string("Activation of layer ") + std::to_string(i)).c_str();
            cv::namedWindow(windowname, CV_WINDOW_NORMAL);
            cv::imshow(windowname, mat);
        }
    }
    cv::waitKey(1);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void visualizeLastLayerActivation(const tiny_cnn::network<tiny_cnn::sequential> &_net)
{
    tiny_cnn::image<unsigned char> img = _net[_net.depth()-1]->output_to_image(); // visualize activations of recent input
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
cv::Mat tinyimage2mat(const tiny_cnn::image<T> &_image)
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
        CV_Error(cv::Error::StsBadArg, "Error in __random_shuffle(): input vectors have different sizes!");
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
void __unskew(const std::vector<T1> &vraw, const std::vector<T2> &vlabel, std::vector<T1> &_outraw, std::vector<T2> &_outlabel)
{
    if(vraw.size() != vlabel.size()) {
        CV_Error(cv::Error::StsBadArg, "Error in __unskew(): input vectors have different sizes!");
        return;
    }

    std::vector<T2> _templabel = vlabel;
    std::sort(_templabel.begin(), _templabel.end());
    auto _last = std::unique(_templabel.begin(), _templabel.end());

    size_t uniquelabels = _last - _templabel.begin();

    std::vector<bool> vlogic;
    for(size_t i = 0; i < uniquelabels; i++)
        vlogic.push_back(false);

    for(size_t i = 0; i < vraw.size(); i++) {

        for(size_t j = 0; j < vlogic.size(); j++) {
            if((vlabel[i] == j) && (vlogic[j] == false)) {
                vlogic[j] = true;
                _outlabel.push_back(vlabel[i]);
                _outraw.push_back(vraw[i]);
            }
        }
        bool check = true;
        for(size_t j = 0; j < vlogic.size(); j++)
              check = check && vlogic[j];
        if( check == true )
            for(size_t j = 0; j < vlogic.size(); j++)
                  vlogic[j] = false;
    }
}

template<typename T>
void __subsetdata(const std::vector<T> &_vin, int _mod, std::vector<T> &_vbig, std::vector<T> &_vsmall)
{
    for(size_t i = 0; i < _vin.size(); i++) {
        if((i % _mod) == 0)
            _vsmall.push_back(_vin[i]);
        else
            _vbig.push_back(_vin[i]);
    }
}

//-------------------------------------------------------------------------------------------------------------------------
TextSegmentConvNet::TextSegmentConvNet()
{}
//-------------------------------------------------------------------------------------------------------------------------
void TextSegmentConvNet::initPretrainedWeights()
{
    // Pretrained weights vector
    tiny_cnn::float_t _weights[] = {
        -1.6169357190442144e+00, -1.0725335296983469e+00,
            -1.1957506751744684e+00, 7.1865706019004194e-01,
            9.9373217319440155e-01, -1.6841600781431854e+00,
            3.7870064958456329e-01, 5.6470341349307440e-01,
            1.4149124543068806e-02, 4.8872394282792869e-01,
            -1.7027096326231128e+00, -6.3357007359732831e-01,
            -7.1952926397926165e-01, 1.0439920420953765e+00,
            5.8621743666522319e-01, 9.2408210024022788e-02,
            1.4153085005618271e-01, 1.3426268811450751e-01,
            8.9388797660640207e-01, 3.2682301692179910e-01,
            1.8637980344638352e-01, 1.1614898355238740e-01,
            9.2198946047619024e-02, -1.2132265696813682e-01,
            2.3166722719686367e-01, 1.4134866092444531e+00,
            -4.1004446302960358e-02, -8.4961717588767338e-01,
            -4.8130830815769132e-01, -3.3541917400223074e-02,
            7.4265329490421639e-01, 4.4805266362856305e-01,
            -6.7656720304652737e-02, -7.6015113626738828e-01,
            -2.4231433699236996e-01, 2.5462434148843593e-01,
            5.7529283210360771e-01, -2.0106271014572196e-02,
            -2.1639474821576229e-01, -7.5370612381641163e-03,
            3.3659401978127917e-01, -3.3689195222844630e-02,
            -7.2624131786225710e-01, 6.3101211657461354e-01,
            1.4543255815176186e+00, -1.4468256079409549e+00,
            -3.2516205228746919e+00, -2.6687818067689810e+00,
            -5.6642184624962866e-01, 2.0917056217830301e+00,
            -1.7023358758404605e+00, -2.3628496326620255e-01,
            -4.9226348626998684e-01, -4.9127252358647161e-01,
            -8.9962409758723305e-01, -7.0478629217320643e-01,
            8.0170837110876525e-01, 1.0084390322284602e+00,
            9.3827861870018359e-01, 1.7312833312661668e-01,
            -5.0804072815972823e-01, 1.3488915632622271e+00,
            1.4523852717372885e+00, 1.2951016245205531e+00,
            -4.9624674685862824e-01, -1.1428941409605291e-01,
            1.2914509407355985e+00, 1.8845452037561314e+00,
            1.1915440091260192e+00, -8.0487835466943380e-01,
            -1.3456290325960276e+00, -2.1603532596199654e-01,
            2.7860541361357738e-01, 1.4039254827144498e-01,
            -1.0113880070067065e+00, 6.6765416934978405e-01,
            1.2069593306845023e-02, 2.8078092845568464e-01,
            -9.6843860715217767e-01, -1.8635898483845967e-01,
            6.6690545989643601e-01, 1.7067707594098783e-01,
            -2.7305747975986766e-01, -2.6118729640529476e-01,
            -4.7925737312609164e-02, 1.3810117752565337e-01,
            -1.1165637245726857e-01, -3.1365870709397781e-01,
            3.8150918158328345e-03, 2.4141328846685246e-02,
            1.4678976416086473e+00, -7.4803073495412528e-02,
            4.9947355394881193e-01, -9.2753604032418013e-01,
            1.2435825257308152e-01, 4.2396239000359354e-01,
            9.4749615452050540e-01, 1.4043271428460642e+00,
            -4.1005413261672902e-01, 9.9565237472836798e-01,
            1.5659068016026816e+00, 8.7550167727414163e-01,
            1.1044897219855403e+00, 6.1979383074339345e-01,
            1.3661166448186163e-01, -1.4467309383166010e-01,
            -2.3078072796487784e-01, -3.6661632485781981e-01,
            -4.9469655567313822e-02, -8.6035360750541812e-01,
            4.3502135283832760e-02, -3.7735869030549624e-01,
            -5.7783407836268963e-01, 1.2962188718995382e-01,
            -2.4025995772859660e-01, 8.7634458743927479e-01,
            -7.8652306905901914e-02, -4.1066845292255710e-01,
            4.3504932403753871e-01, 4.4689104455436757e-01,
            -7.4762793868078259e-01, -1.0760723828949288e+00,
            -2.0336228018316165e+00, -1.2747463629413225e+00,
            -9.7844918441810125e-01, -1.2755066521705456e+00,
            -1.6974832163269804e+00, -2.3197481636243049e+00,
            -8.9739649525173193e-01, 2.0065025544528134e-01,
            1.5413786200154916e-02, -1.8429457305620597e-01,
            -1.0337313579033618e+00, 1.6779450854365965e-01,
            1.2529987348376461e+00, -1.2994015956803699e+00,
            -7.6181561289336963e-01, -5.1443897587443776e-01,
            -3.6703771597724111e-01, 4.7600567708629044e-01,
            -7.5060520586017110e-01, -3.8735146174898710e-01,
            -9.7302850382999712e-02, 1.6654091393583911e-02,
            -4.5444868191826893e-01, -1.1734383340415522e-01,
            1.7497511219502504e+00, 1.4588984541275412e+00,
            1.0704626541303717e+00, 4.6451460661698446e-01,
            1.0295331593408694e-01, -3.1857440126550833e+00,
            -2.4376919597672271e+00, -1.0894482451801131e+00,
            6.8487564555237979e-01, -3.5109558927505968e-01,
            -1.8141790071426782e+00, -1.0946764788752801e+00,
            1.1492550436169000e-01, 1.1118656681621784e+00,
            -4.7280773524961567e-01, 2.9598459621700535e-01,
            1.6247761275106407e-01, 4.9552612031443510e-01,
            -3.3101330871068591e-01, -1.3575055672082306e+00,
            -4.6568151185362355e-01, -1.6414063857188069e-01,
            -2.6573406649269149e-01, -6.9935254973532335e-01,
            2.1029305851981652e+00, 1.2513010499454598e+00,
            1.3021503826250289e+00, 1.2012426236545410e+00,
            5.9695910151733889e-01, 1.5764364826334492e+00,
            9.5272070564235456e-01, 1.3465308768778501e+00,
            2.1573425395285346e+00, 1.4036311241723036e+00,
            -6.3151999163215830e-02, -6.5379836957932957e-01,
            3.7274494728987050e-02, 5.2619918325974284e-02,
            -7.1448119486586392e-01, 8.0534538901475927e-01,
            3.8218978569903372e-01, 2.4639547104240722e-01,
            -1.4795203228255052e-01, -1.1483048729931811e+00,
            5.0250434161123081e-01, -3.6799484380578112e-01,
            -8.9404158440814074e-01, -1.6025760870956696e+00,
            -1.7769531081995313e+00, 8.2901132002775701e-01,
            -9.4385201721889878e-01, -1.5060780446644975e+00,
            -2.1011943078313728e+00, -1.4908314540620480e+00,
            4.8029485696294849e-01, 8.2777750892490631e-01,
            7.9602599665581020e-01, -1.0694426670381607e+00,
            4.0371952988993398e-01, 5.0571791435406055e-01,
            1.0955803408560894e+00, 7.4418774444364300e-01,
            -1.2547480751301279e-01, 1.9877472552198808e-01,
            -3.2381963614733889e-01, 2.8643425165985154e-01,
            -2.2336197663397581e-01, -8.3331963734338294e-02,
            -4.8953453277028736e-01, 1.9117804901834945e-01,
            8.8687506853410855e-02, 1.1951107067078050e-02,
            4.0640696287693311e-02, -2.4334968551917951e-01,
            -2.5371982243574126e-01, -6.9128370927353938e-02,
            1.3060431822562885e-01, -2.9809395427287300e-01,
            2.0062652502861114e-01, 3.9052500956833193e-02,
            -4.3367422121697519e-01, 3.2334522320723741e-01,
            -2.4402735168853590e-01, 2.1051436359102005e-02,
            -2.5948063131262272e-02, 2.6517414689380875e-01,
            -2.9909196416535277e-01, 9.1830854712519616e-02,
            2.8914853032886884e-01, -2.1068958212494152e-01,
            -3.3919417692963599e-01, 3.3327763753297041e-01,
            -3.0682058442979132e-01, 1.7939977286720737e-01,
            -1.7228114024776173e-01, 3.4793465644987693e-03,
            -2.2989823895537817e-01, 3.8004789873030392e-01,
            3.9852423633248008e-01, -3.0101285436340536e-01,
            6.1421112522197785e-02, 2.6481909826522559e-03,
            2.8846718678754801e-01, -2.5978265398760436e-01,
            1.0307808609196663e-01, -2.5390450789506719e-01,
            8.2494948270169441e-02, -2.8923481086311431e-01,
            -7.5554079984153671e-02, -4.5659097354421235e-01,
            1.6706742909820832e-01, -1.8055348978862223e-01,
            3.2178860203512899e-01, 5.0478581729108345e-02,
            -3.3569306488583088e-03, -3.7424223478982427e-01,
            -2.4591998903260701e-01, -1.1442848330234717e-01,
            -2.4091383313957215e-01, 1.0712402081298941e-01,
            -1.0695899570200809e-01, 3.3327254144050289e-01,
            -7.5162987720914301e-02, -4.3100763566148620e-01,
            1.3654527694978735e-01, 2.5636535108261754e-01,
            -3.9661180644324928e-01, -1.9764851261103240e-02,
            5.1060845941949595e-01, -6.0902086264474198e-01,
            -4.9597475487343022e-02, -3.6990015845027596e-01,
            -1.6195371308184217e-01, -3.0613649050703812e-02,
            4.9120114503990904e-02, -2.0328818905459906e-01,
            8.3187656505356436e-02, -2.2956368978326799e-02,
            8.4597208224069215e-02, -4.9241240280645676e-01,
            -6.7222583285952528e-02, -3.5886060055727265e-01,
            3.6065627750932455e-01, 4.7651114477463299e-02,
            2.4190553810333124e-01, -5.0725779376224145e-01,
            2.2713361480755889e-01, -3.1852627478312712e-01,
            4.8253925606351100e-01, -4.5691320183767470e-01,
            1.1425572276793683e-01, -6.1095846724436498e-01,
            3.7208866652196491e-02, -3.4295539788602741e-01,
            1.6214766779257991e-01, -2.3116840716031417e-01,
            3.5405619404830169e-01, -1.2105249780393702e-01,
            2.8854602382076044e-01, -3.2049863533417405e-01,
            -6.0323671216410994e-03, -3.7801508667077499e-01,
            2.5691682944310468e-01, -2.9210423406148006e-01,
            4.4869639538447936e-01, -5.3542448682833776e-01,
            3.0818895651552541e-01, -6.5489732934599254e-02,
            1.8191932402243724e-01, 1.5576738798759174e-03,
            -1.7495987500538016e-01, 1.6246485064902969e-01,
            -9.5769359989769307e-02, 9.9977856599757442e-02,
            -4.3094828650181438e-01, 2.3704170076721431e-01,
            -2.8311002899869370e-01, 1.7693483619374625e-01,
            -3.7384707542841111e-01, 9.7287081854169885e-02,
            -3.1907306059979212e-01, 1.6451593169456491e-01,
            -5.2144127174718280e-01, 3.8629039783798924e-01,
            -1.7063610871439139e-01, 2.9008246607078653e-01,
            -2.7281893119465439e-01, 2.7730241621835250e-01,
            -2.4385911273082347e-01, 1.7588937463214274e-01,
            -2.7550491526776794e-01, 8.5622721858167009e-02,
            -1.0300489146448341e-01, 1.7860907568321585e-01,
            -2.1652377556894309e-01, 1.8654785421691147e-01,
            1.0219082401336632e-01, -8.7579056275908979e-02,
            -3.2406094198604840e-01, 2.4349496486268236e-02,
            -2.7560274799756285e-01, -9.2783736687879939e-02,
            -1.5682239997850950e-01, 5.7104935149531820e-02,
            -3.0942102535616850e-01, 2.7898917227120251e-01,
            2.7025175578780036e-01, -2.7025175578779981e-01 };
    // Load weights
    setInputSize( cv::Size(7,7) );
    setInputChannels(1);
    setOutputChannels(2);
    initNet(getInputSize(), getInputChannels(), getOutputChannels() ,_weights);

}
//-------------------------------------------------------------------------------------------------------------------------
tiny_cnn::network<tiny_cnn::sequential> TextSegmentConvNet::__createNet(const cv::Size &size, int inchannels, int outchannels)
{
    network<sequential> _net;
    cnn_size_t _width = static_cast<cnn_size_t>(size.width),
               _height = static_cast<cnn_size_t>(size.height),
               _kernels = 8;

    _net << convolutional_layer<relu>(_width, _height, 5, static_cast<cnn_size_t>(inchannels), _kernels)
         << fully_connected_layer<softmax>((_width-4)*(_height-4)*_kernels, static_cast<cnn_size_t>(outchannels));

    return _net;
}
//-------------------------------------------------------------------------------------------------------------------------
} // end of the segnet namespace



