#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mkl_dnn_layers.hpp"
#include "caffe/layers/softmax_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_softmax_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MKL_DNNSoftmaxLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MKL_DNNSoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 1000, 1, 1)),  // simple case TODO: batch support
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MKL_DNNSoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MKL_DNNSoftmaxLayerTest, ::testing::Types<CPUDevice<float> >);

TYPED_TEST(MKL_DNNSoftmaxLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MKL_DNNSoftmaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        Dtype sum = 0;
        for (int j = 0; j < this->blob_top_->channels(); ++j) {
          sum += this->blob_top_->data_at(i, j, k, l);
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        Dtype scale = 0;
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          scale += exp(this->blob_bottom_->data_at(i, j, k, l));
        }
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + 1e-4,
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
          EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - 1e-4,
              exp(this->blob_bottom_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

#if 0  // TODO
TYPED_TEST(MKL_DNNSoftmaxLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MKL_DNNSoftmaxLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
#endif


}  // namespace caffe
