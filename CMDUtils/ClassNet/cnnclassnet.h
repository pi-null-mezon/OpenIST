#ifndef CNNCLASSNETH
#define CNNCLASSNETH

#include <tiny_cnn/tiny_cnn.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace tiny_cnn;

enum ImageResizeMethod {CropAndResizeFromCenter, PaddZeroAndResize};
/**
 * @brief The base abstract class
 * @note Derived classes should override __initNet() method
 */
class CNNClassificator
{
public:    
    CNNClassificator();
    virtual ~CNNClassificator();
    /**
     * @brief train - prepares data and starts training for particular number of epoch with desired minibatch
     * @param _vvis - raw images vector, could have arbitrary quantity of channels and 8-bit or 16-bit per channel depth
     * @param _vlabel - labels for training
     * @param _epoch - number of training iterations (one iteration is performed on whole training data set)
     * @param _minibatch - how many samples should be enrolled before parameters of the network will be updated (select in range from 1 to 16), the greater value is used the smoothed loss function will be that in general prevents local minimum jam
     * @param _t2cprop - divisor for data subsampling into training and control sets, for the instance if _t2cprop eqauls 5 it means that each 5-th sample goes to control set
     */
    void train(cv::InputArrayOfArrays _vraw, const std::vector<label_t> &_vlabel, int _epoch, int _minibatch, int _t2cprop);
    /**
     * @brief update - prepares data and runs another training session, note that update() differs from train() by weight initialization method. The train() method use random weights seeding whereas update() preserves weights that has been learned from previous train() or update() or loaded by load()
     * @param _vvis - same as in train()
     * @param _vseg - same as in train()
     * @param _epoch - same as in train()
     * @param _minibatch - same as in train()
     * @param _t2cprop - same as in train()
     */
    void update(cv::InputArrayOfArrays _vraw, const std::vector<label_t> &_vlabel, int _epoch, int _minibatch, int _t2cprop);

    void save(const char *filename) const;
    bool load(const char *filename);

    cv::Size getInputSize() const;
    void setInputSize(const cv::Size &_size);

    int getInputChannels() const;
    void setInputChannels(int _value);

    int getOutputChannels() const;
    void setOutputChannels(int _value);

    void setImageResizeMethod(ImageResizeMethod method);
    ImageResizeMethod getImageResizeMethod() const;

    label_t predict(const cv::Mat &image) const;

private:
    void __train(cv::InputArrayOfArrays _vraw, const std::vector<label_t> &_vlabel, int _epoch, int _minibatch, int _t2cprop, bool preservedata);
    virtual tiny_cnn::network<tiny_cnn::sequential> __createNet(const cv::Size &size, int inchannels, int outchannels);

    mutable tiny_cnn::network<tiny_cnn::sequential> m_net;
    ImageResizeMethod m_irm = PaddZeroAndResize;
    cv::Size m_inputsize;
    int m_inputchannels;
    int m_outputchannels;   
};

/**
 * @brief __cropresize - crops and then resizes ROI of the input image with size proportion
 * @param input - imput image
 * @param size - target size of output image
 * @return resized image
 * @note inscribed rect ROI is returned
 */
cv::Mat __cropresize(const cv::Mat &input, const cv::Size size);

/**
 * @brief __propresize - crops and then
 * @param input - input image
 * @param size - target size of output image
 * @return resized image
 * @note described rect ROI is returned
 */
cv::Mat __propresize(const cv::Mat &input, const cv::Size size);

/**
 * @brief __mat2vec_t - converts cv::Mat into tiny_cnn::vec_t
 * @param img - input image
 * @param targetSize - target size after conversion
 * @param resizeMethod - how input image should be resized to the targetSize
 * @param targetChannels - desired quantity of the output channels
 * @param min - output minimum values
 * @param max - output maximum values
 * @return image in vector format that is used by tiny_cnn::network::fit()/train()
 */
tiny_cnn::vec_t __mat2vec_t(const cv::Mat &img, const cv::Size targetSize, ImageResizeMethod resizeMethod, int targetChannels, double min=-1.0, double max=1.0);

/**
 * @brief visualizeActivations - draws activations of all layers in a bunch of windows
 * @param _net - network to parse
 */
void visualizeActivations(const tiny_cnn::network<tiny_cnn::sequential> &_net);

/**
 * @brief visualizeLastLayerActivation - self explained
 * @param _net - network to parse
 */
void visualizeLastLayerActivation(const tiny_cnn::network<tiny_cnn::sequential> &_net);

template<typename T>
/**
 * @brief tinyimage2mat - converts tiny_cnn::image to cv::Mat image
 * @param _image - image that should be converted
 * @return converted image
 */
cv::Mat tinyimage2mat(const tiny_cnn::image<T> &_image);

template <typename Iterator1, typename Iterator2>
/**
 * @brief __random_shuffle - performs random shuffle simultaneously for two vectors (raw images and label images for the instance)
 * @param v1first - begin of the first vector
 * @param v1last - end of the first vector
 * @param v2first - begin of the second vector
 * @param v2last - end of the second vector
 */
void __random_shuffle (Iterator1 v1first, Iterator1 v1last, Iterator2 v2first, Iterator2 v2last);
/**
 * Filters data in a particular way to make equal (or almost equal, i.e. +-1) size of each label subset
 * Optionaly can return quantity of the unique labels in data by _ulabels argument
 */
template<typename T1, typename T2>
void __unskew(const std::vector<T1> &vraw, const std::vector<T2> &vlabel, std::vector<T1> &_outraw, std::vector<T2> &_outlabel, size_t *_ulabels=0);
/**
 * Divides data into two subset for training and for validation purposes
 */
template<typename T>
void __subsetdata(const std::vector<T> &_vin, int _mod, std::vector<T> &_vbig, std::vector<T> &_vsmall);


#endif // CNNCLASSNETH
