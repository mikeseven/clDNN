#ifdef USE_NEURALIA
#include <algorithm>
#include <vector>

#include "caffe/layers/neuralia_layers.hpp"
#include "neural.h"

using namespace neural;

namespace caffe {
template <> NeuraliaReLULayer<double>::~NeuraliaReLULayer() {NOT_IMPLEMENTED;}
template <> void NeuraliaReLULayer<double>::LayerSetUp(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void NeuraliaReLULayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {NOT_IMPLEMENTED;}
template <> void NeuraliaReLULayer<double>::Backward_cpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {NOT_IMPLEMENTED;}

template <typename Dtype>
NeuraliaReLULayer<Dtype>::~NeuraliaReLULayer() {}

template <typename Dtype>
void NeuraliaReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

//  CHECK_EQ(top[0]->shape(), bottom[0]->shape());

  vector<unsigned> sizes;
  for (auto d : bottom[0]->shape())
      sizes.push_back(d);
  
  // TODO: change format?
  bottom_data_  = memory::create({engine::cpu, memory::format::yxfb_f32, sizes});
  top_data_     = memory::create({engine::cpu, memory::format::yxfb_f32, sizes});
  bottom_diff_  = memory::create({engine::cpu, memory::format::yxfb_f32, sizes});
  top_diff_     = memory::create({engine::cpu, memory::format::yxfb_f32, sizes});
    
  reluFwd_ = relu::create({engine::reference, top_data_, bottom_data_, negative_slope});
  reluBwd_ = relu_backward::create({engine::reference, {bottom_diff_}, {top_diff_, bottom_data_}, negative_slope});
}

template <typename Dtype>
void NeuraliaReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  void* bottom_data = (void*)bottom[0]->prv_data();
  void* top_data = NULL;

  if (bottom_data) {
    top_data = top[0]->mutable_prv_data();
  }
  else {
    DLOG(INFO) << "Using cpu_data in NeuraliaReLULayer.";
    bottom_data = (void*)bottom[0]->cpu_data();
    top_data = top[0]->mutable_cpu_data();
  }

  execute({bottom_data_(bottom_data), top_data_(top_data), reluFwd_});
}

template <typename Dtype>
void NeuraliaReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    void* top_diff = (void*)top[0]->prv_diff();
    void* bottom_data = NULL;
    void* bottom_diff = NULL;

    if (top_diff && bottom[0]->prv_data()) {
      bottom_data = (void*)bottom[0]->prv_data();
      bottom_diff = (void*)bottom[0]->mutable_prv_diff();

      if (NULL == bottom_data)
        LOG(FATAL) << "bottom_data is NULL";
    } else {
      DLOG(INFO) << "Using cpu_data in NeuraliaReLULayer.";
      top_diff = (void*)top[0]->cpu_diff();
      bottom_data = (void*)bottom[0]->cpu_data();
      bottom_diff = (void*)bottom[0]->mutable_cpu_diff();
    }

    execute({ top_diff_(top_diff), bottom_data_(bottom_data), bottom_diff_(bottom_diff), reluBwd_});
  }
}

#ifdef CPU_ONLY
STUB_GPU(NeuraliaReLULayer);
#endif

INSTANTIATE_CLASS(NeuraliaReLULayer);
}  // namespace caffe
#endif //#ifdef USE_NEURALIA