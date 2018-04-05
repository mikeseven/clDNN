/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "common/common_tools.h"
#include "file.h"
#include <string>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/reorder.hpp>
#include <api/CPP/convolution.hpp>
#include <api/CPP/pooling.hpp>
#include <api/CPP/concatenation.hpp>
#include <api/CPP/softmax.hpp>

using namespace cldnn;


#define CREATE_CALIBRATION_OBJECTS(cur, prev)  \
    auto& cur_w = cur_weights.mem;                                                                                              \
    apply_calibration_on_weights<float>(cur_w, prev_calib);                                                                     \
    auto cur_w_qf = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, cur_w.get_layout().size.batch[0], 1, 1 } });  \
    auto cur_o_qf = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, (int32_t)cur_calib.size(), 1, 1 } });         \
    set_values(cur_o_qf, cur_calib);                                                                                            \
    quantize_weights<float>(cur_w, cur_w_qf);                                                                                   \
    auto cur_w_int = create_int8_weights(engine, cur_w);                                                                        \
    auto cur_weigths_int = data("cur_weights_int", cur_w_int);                                                                  \
    auto cur_weights_qf  = data("cur_w_qf", cur_w_qf);                                                                          \
    auto cur_output_qf   = data("cur_o_qf", cur_o_qf); 


template<typename T>
void set_values(const cldnn::memory& mem, std::vector<T>& args) {
    auto ptr = mem.pointer<T>();

    auto it = ptr.begin();
    for (auto x : args)
        *it++ = x;
}

std::vector<float> conv10_calib = {
    1.711879f,  4.174624f,  2.023904f,  2.7478f,  2.580315f,  2.153681f,  1.637389f,  1.66968f,  1.48105f,  1.27478f,  1.124511f,  2.454106f,  2.424821f,  1.727891f,  2.432074f,  3.599641f,  2.757123f,  2.413302f,  3.97846f,  1.774672f,  1.900842f,  1.329843f,  1.499631f,  1.262112f,  1.187604f,  1.144784f,  2.294747f,  2.410439f,  1.179333f,  2.145723f,  1.390828f,  1.02574f,  1.186916f,  1.616547f,  2.073469f,  1.616547f,  1.676568f,  2.175589f,  1.468208f,  1.52782f,  2.596802f,  2.816352f,  2.162851f,  1.852325f,  3.067166f,  1.835592f,  2.560804f,  1.819158f,  1.645344f,  1.450393f,  1.814286f,  1.701843f,  3.389487f,  3.088146f,  1.811052f,  2.442308f,  2.457074f,  3.272142f,  1.909774f,  1.835592f,  1.735269f,  1.51755f,  2.496314f,  2.221977f,  2.353211f,  2.100257f,  1.972816f,  1.779335f,  1.832281f,  3.432432f,  3.568037f,  2.289575f,  2.925843f,  2.166311f,  2.84993f,  3.21519f,  2.307778f,  2.107884f,  2.603457f,  3.114172f,  1.874539f,  2.909088f,  1.953846f,  2.205098f,  2.033015f,  1.757785f,  1.933397f,  1.380435f,  3.160187f,  2.215921f,  2.857947f,  3.762963f,  5.972086f,  3.470533f,  2.003945f,  3.420875f,  2.673684f,  1.290159f,  2.127749f,  1.745704f,  2.379391f,  1.657423f,  1.018036f,  1.043653f,  1.115865f,  0.747059f,  0.972249f,  3.628571f,  2.981655f,  3.2512f,  1.299233f,  3.580612f,  1.974733f,  2.588535f,  3.769944f,  2.26659f,  2.500921f,  2.166311f,  1.465032f,  2.432074f,  3.121352f,  3.282714f,  2.270391f,  2.351852f,  2.913258f,  2.837989f,  3.808804f,  5.22365f,  3.888989f,  4.641915f,  1.497421f,  2.294747f,  5.186976f,  2.200323f,  1.66694f,  1.847273f,  2.635538f,  2.217128f,  2.575412f,  1.125757f,  1.320338f,  2.074526f,  2.181426f,  3.412254f,  2.942792f,  1.833935f,  1.763889f,  2.382181f,  3.3559f,  1.525526f,  1.35557f,  1.031992f,  2.915352f,  1.907981f,  2.878187f,  2.57868f,  1.996071f,  1.617834f,  1.002463f,  0.911211f,  1.195294f,  0.937269f,  0.817377f,  1.225573f,  1.038855f,  1.343915f,  0.776165f,  0.832105f,  1.087794f,  0.934683f,  1.136465f,  0.769697f,  0.709497f,  1.114035f,  1.0299f,  2.12441f,  0.980695f,  1.000985f,  1.364674f,  1.165138f,  0.897527f,  1.177289f,  1.27478f,  1.449358f,  1.283639f,  1.869365f,  1.077413f,  1.933397f,  1.039382f,  1.171857f,  1.499631f,  1.331586f,  1.442158f,  1.357381f,  1.717667f,  1.920605f,  1.526672f,  1.594976f,  1.253541f,  1.811052f,  2.059806f,  1.779335f,  1.414057f,  1.265255f,  2.605128f,  1.066107f,  0.801262f,  0.79375f,  0.888112f,  0.976923f,  1.248918f,  1.297573f,  1.058333f,  1.229274f,  1.713322f,  1.08954f,  1.559478f,  1.996071f,  1.188304f,  1.277987f,  1.331586f,  1.044193f,  0.931256f,  1.264462f,  1.012452f,  1.09778f,  1.972816f,  1.010439f,  1.820789f,  2.360046f,  1.14221f,  1.077413f,  1.210965f,  1.682119f,  1.473532f,  0.974113f,  1.049587f,  1.320338f,  1.455587f,  1.974733f,  1.079124f,  1.117712f,  1.344805f,  1.359197f,  1.197402f,  1.585023f,  1.000985f,  1.213129f,  1.124511f,  1.033571f,  1.075697f,  0.869863f,  1.671053f,  0.726233f,  1.476744f,  1.290159f,  1.350166f,  1.005439f,  1.407202f,  1.285262f,  0.865417f,  1.451429f,  1.605055f,  0.867635f,  0.909579f,  0.912029f,  0.975985f,  0.937269f,  0.926162f,  0.747609f,  0.803162f,  0.814755f,  0.858833f,  0.652537f,  0.769114f,  1.047423f,  0.792512f,  0.7373f,  0.73041f,  1.104944f,  0.740525f,  1.974733f,  1.542901f,  2.237885f,  1.414057f,  1.516418f,  0.689749f,  1.14221f,  1.176601f,  1.21749f,  0.790661f,  1.057778f,  1.644013f,  0.835526f,  0.866894f,  1.140292f,  0.721591f,  0.845962f,  0.875862f,  0.787597f,  6.27644f,  4.610319f,  2.770276f,  2.855935f,  3.449915f,  2.485624f,  1.893756f,  2.127749f,  3.92278f,  3.167572f,  1.862511f,  1.413074f,  1.965184f,  3.39515f,  3.333876f,  3.017071f,  2.303855f,  2.115563f,  4.132178f,  3.571178f,  2.994838f,  3.026057f,  2.962099f,  3.130971f,  3.014837f,  4.187536f,  1.955727f,  2.572152f,  2.427718f,  2.731183f,  2.211099f,  1.724958f,  2.264065f,  1.31521f,  2.260289f,  1.128257f,  1.148023f,  1.21749f,  2.360046f,  1.605055f,  1.48105f,  1.165138f,  1.07116f,  1.25509f,  1.127636f,  1.27f,  1.075132f,  1.031472f,  0.826688f,  0.912848f,  1.322917f,  1.435028f,  2.106789f,  1.216766f,  1.083729f,  0.983543f,  1.266833f,  1.65069f,  1.329843f,  1.316915f,  1.126386f,  1.52782f,  1.017017f,  2.40047f,  1.316915f,  0.976923f,  1.423966f,  1.739726f,  1.682119f,  1.948226f,  1.343027f,  1.735269f,  1.180703f,  1.337722f,  1.790308f,  2.290868f,  1.676568f,  1.96139f,  2.192017f,  1.275581f,  1.331586f,  1.266037f,  0.923636f,  1.440113f,  1.787159f,  1.716216f,  1.727891f,  1.064427f,  1.478894f,  1.916981f,  3.104656f,  1.878004f,  2.832053f,  3.021558f,  2.309091f,  2.18612f,  2.458558f,  1.454545f,  1.621708f,  2.716578f,  1.753236f,  1.140927f,  3.482429f,  1.644013f,  3.069486f,  4.170347f,  2.830084f,  2.49478f,  2.836005f,  4.117521f,  2.46303f,  2.401891f,  1.987285f,  2.098088f,  2.384977f,  3.290684f,  1.560676f,  2.092688f,  1.448325f,  1.920605f,  1.803017f,  2.826147f,  2.116667f,  2.066089f,  1.824057f,  3.350367f,  2.800824f,  2.940666f,  1.630819f,  2.007905f,  1.809439f,  1.980507f,  1.298403f,  2.234193f,  1.389877f,  1.835592f,  3.941798f,  2.618557f,  1.586261f,  2.615187f,  2.199134f,  1.484295f,  3.704644f,  2.657944f,  1.624301f,  1.850638f,  1.994112f,  1.429979f,  2.164004f,  3.561783f,  2.24903f,  2.115563f,  1.654723f,  2.562421f,  1.53012f,  3.179969f,  2.217128f,  1.539394f,  2.57868f,  2.505549f,  3.590106f,  2.360046f,  2.663172f,  1.275581f,  2.983847f,  1.763889f,  3.165109f,  1.388927f,  5.083172f,  1.479971f,  2.932179f,  1.597484f,  3.423753f,  2.78738f,  2.580315f,  2.490196f,  1.293444f,  1.717667f,  2.173262f,  3.306749f,  1.605055f,  2.557581f,  3.048759f,  1.867647f,  4.302809f,  4.758782f,  2.134454f,  1.408177f,  2.177921f,  2.606797f,  2.635538f,  2.307778f,  2.404734f,  3.272142f,  2.272931f,  2.026931f,  1.350166f,  3.992141f,  2.397638f,  1.995089f,  3.118952f,  2.35594f,  1.953846f,  3.001477f,  1.675185f,  2.426266f,  3.2f,  1.174566f,  1.886722f,  2.314351f,  3.684491f,  2.062944f,  2.320957f,  1.773124f,  2.427718f,  1.763889f,  2.521092f,  1.628205f,  1.84058f,  2.038114f,  2.497846f,  1.221882f,  1.229274f,  1.855708f,  1.963285f,  3.347611f,  2.182599f,  2.934293f,  2.630418f,  3.048759f,  2.24903f,  2.323611f,  2.606797f,  2.413302f,  1.860806f,  2.354577f,  3.126154f,  1.100158f,  4.881686f,  1.528969f,  1.820789f,  3.988217f,  2.851927f,  2.201517f,  2.267857f,  4.127987f,  2.061896f,  2.749662f,  2.007905f,  2.727517f,  1.700418f,  2.128861f,  1.353764f,  3.776952f,  4.172485f,  2.060852f,  2.642393f,  2.022895f,  1.308435f,  1.642684f,  2.160551f,  4.835222f,  2.616868f,  1.684909f,  4.033744f,  2.944928f,  1.394647f,  3.184953f,  1.727891f,  2.234193f,  2.086242f,  3.512528f,  2.768392f,  3.46167f,  3.121352f,  2.330275f,  2.151401f,  3.497418f,  2.659686f,  2.593488f,  3.312139f,  4.376956f,  3.648115f,  2.878187f,  2.888412f,  1.576416f,  2.014872f,  1.735269f,  1.773124f,  1.540561f,  1.566692f,  2.496314f,  5.250646f,  1.48538f,  2.552764f,  2.785466f,  1.711879f,  3.309446f,  2.642393f,  3.243412f,  2.119977f,  3.470533f,  1.319481f,  1.819158f,  1.722034f,  2.135574f,  2.745946f,  3.2f,  3.172518f,  2.613502f,  2.256522f,  2.387779f,  1.730835f,  2.892523f,  3.238243f,  1.433004f,  1.972816f,  3.039637f,  1.591229f,  2.196757f,  3.202518f,  2.749662f,  1.305074f,  2.654471f,  4.586907f,  3.941798f,  1.722034f,  3.72844f,  2.611825f,  1.597484f,  2.482588f,  1.401379f,  2.446716f,  2.391993f,  3.044191f,  3.248597f,  2.076646f,  1.153235f,  2.110073f,  1.605055f,  3.694545f,  2.038114f,  2.476536f,  1.362844f,  2.511743f,  2.149126f,  3.197478f,  2.460048f,  2.138947f,  2.500921f,  1.507418f,  2.187298f,  2.618557f,  1.45977f,  1.722034f,  3.497418f,  2.915352f,  3.606029f,  1.538229f,  2.481074f,  2.182599f,  3.386667f,  4.277895f,  3.628571f,  2.680739f,  2.923741f,  2.247788f,  1.487555f,  4.358181f,  3.494406f,  1.854015f,  4.059934f,  2.383575f,  2.944928f,  1.230024f,  1.693333f,  1.633441f,  1.982439f,  1.188994f,  1.386085f,  1.46082f,  2.076646f,  5.274503f,  1.114035f,  1.285262f,  1.996071f,  2.855935f,  1.456631f,  1.400414f,  3.59646f,  1.180023f,  1.346587f,  2.151401f,  3.378217f,  4.123792f,  2.458558f,  2.543179f,  4.291439f,  3.657961f,  1.02574f,  1.722034f,  2.632124f,  1.372046f,  1.640032f,  1.417015f,  3.533913f,  5.06418f,  0.809562f,  2.101344f,  2.309091f,  3.055639f,  3.107034f,  1.883225f,  1.441135f,  1.929725f,  1.729362f,  1.36193f,  1.963285f,  3.602837f,  1.744206f,  2.890469f,  2.580315f,  1.883225f,  5.447721f,  3.217731f,  2.0f,  2.454106f,  4.485651f,  1.726423f,  2.272931f,  2.166311f,  1.583788f,  1.99705f,  3.691184f,  3.9941f,  2.47503f,  1.717667f,  1.884972f,  2.513294f,  2.84793f,  1.745704f,  3.590106f,  1.616547f,  1.950096f,  3.227955f,  3.369818f,  2.111167f,  3.328415f,  2.647554f,  2.779754f,  2.785466f,  3.467577f,  2.362791f,  1.57764f,  1.804618f,  3.152828f,  3.290684f,  2.637246f,  1.771578f,  1.684909f,  2.22441f,  1.431994f,  3.296022f,  2.372444f,  1.657423f,  2.851927f,  1.899065f,  3.320261f,  1.74271f,  1.360107f,  2.913258f,  1.028857f,  1.686307f,  1.322056f,  3.745617f,  2.251522f,  1.216032f,  4.039761f,  1.777778f,  2.094845f,  5.329842f,  3.312139f,  2.237885f,  2.119977f,  1.525526f,  1.407202f,  1.716216f,  1.57764f,  2.174423f,  1.394647f,  1.291799f,  1.684909f,  2.335632f,  2.880224f,  1.726423f,  2.117768f,  2.033015f,  2.368298f,  3.756007f,  3.389487f,  1.310123f,  3.43824f,  2.350489f,  3.296022f,  1.199528f,  1.061651f,  1.704698f,  2.757123f,  2.00197f,  1.526672f,  1.488645f,  2.025922f,  2.516406f,  1.167816f,  2.240353f,  4.436681f,  3.109407f,  3.900192f,  3.277419f,  1.845595f,  2.49172f,  4.826603f,  3.48542f,  1.842248f,  2.733017f,  2.666667f,  2.08517f,  1.662848f,  2.141201f,  2.119977f,  1.980507f,  1.922422f,  3.464616f,  2.712951f,  1.736752f,  2.362791f,  2.514851f,  1.955727f,  1.79664f,  3.314845f,  3.097561f,  2.406155f,  1.489736f,  1.806222f,  1.390828f,  1.820789f,  2.420486f,  2.088386f,  3.15528f,  1.953846f,  1.200231f,  2.323611f,  0.874355f,  2.394812f,  2.994838f,  1.439093f,  2.173262f,  2.546366f,  1.557088f,  1.312661f,  3.085797f,  2.00197f,  1.35557f,  2.892523f,  3.014837f,  1.408177f,  2.288288f,  3.309446f,  1.874539f,  2.652742f,  1.703269f,  4.789633f,  2.657944f,  2.551159f,  3.586932f,  1.640032f,  1.613979f,  5.099111f,  2.473522f,  1.112201f,  1.704698f,  3.005917f,  4.117521f,  1.642684f,  3.606029f,  3.175f,  1.402346f,  2.97511f,  3.409396f,  1.559478f,  2.652742f,  3.530838f,  2.080899f,  3.602837f,  4.803783f,  2.521092f,  2.113363f,  1.511905f,  1.554705f,  2.452623f,  2.700329f,  1.900842f,  1.361018f,  1.363758f,  3.055639f,  4.13639f,  2.892523f,  3.866788f,  4.365191f,  1.756266f,  1.735269f,  2.722033f,  2.223195f,  1.976654f,  4.092642f,  1.322056f,  2.283146f,  1.254321f,  1.815907f,  2.33832f,  3.762963f,  1.242807f,  2.628719f,  2.904929f,  2.098088f,  1.913371f,  1.654723f,  2.605128f,  1.915174f,  0.967619f,  2.75152f,  1.701843f,  4.323404f,  2.160551f,  2.410439f,  2.419048f,  2.214712f,  1.098378f,  1.777778f,  1.759307f,  3.488407f,  2.5689f,  2.808566f,  2.499385f,  2.722033f,  3.347611f,  2.959938f,  2.517968f,  1.346587f,  2.245304f,  4.938022f,  2.830084f,  2.863985f,  1.413074f,  1.494118f,  3.25641f,  2.027944f,  3.392321f,  2.377997f,  2.925843f,  2.785466f,  2.211099f,  3.026057f,  1.16714f,  1.384196f,  1.35557f,  2.770276f,  3.081118f,  2.155966f,  3.353135f,  2.845938f,  1.538229f,  2.635538f,  3.162642f,  2.18377f,  1.738238f,  1.741217f,  1.277987f,  2.13782f,  3.353135f,  2.159405f,  2.275476f,  2.24903f,  1.828983f,  2.964256f,  4.605103f,  3.97846f,  0.916968f,  1.937083f,  3.274774f,  2.384977f,  2.944928f,  3.235669f,  1.607595f,  2.671924f,  2.17094f,  2.250277f,  1.343027f,  0.874355f,  1.292621f, };
std::vector<float> fire9_concat_calib = {
    0.938135f,  1.254321f,  7.610487f,  0.686951f,  0.0f,  4.161806f,  0.0f,  5.506775f,  11.779656f,  4.178924f,  4.268908f,  0.0f,  3.926564f,  3.375415f,  7.402529f,  3.240829f,  9.769231f,  0.848789f,  4.761566f,  1.915174f,  3.37261f,  10.223879f,  1.179333f,  0.0f,  0.0f,  9.021104f,  6.389937f,  8.887894f,  5.257428f,  6.011834f,  0.0f,  0.651282f,  4.66322f,  2.656209f,  1.854015f,  14.111111f,  0.986408f,  1.606324f,  4.508038f,  6.011834f,  1.540561f,  9.066377f,  1.984375f,  2.215921f,  1.585023f,  0.0f,  0.822006f,  0.0f,  1.998033f,  1.13837f,  0.65718f,  2.015873f,  1.757785f,  25.721519f,  2.307778f,  2.10243f,  1.404285f,  0.824675f,  1.84058f,  1.54173f,  8.055513f,  1.791887f,  6.320358f,  0.0f,  3.165109f,  0.0f,  3.458719f,  2.279302f,  9.65317f,  13.227017f,  0.0f,  4.209215f,  1.082574f,  2.816352f,  22.268474f,  55.011934f,  0.8f,  4.431835f,  2.111167f,  2.114464f,  1.49302f,  0.0f,  0.0f,  0.0f,  0.734104f,  2.116667f,  1.955727f,  0.0f,  0.0f,  8.327869f,  2.21955f,  1.010439f,  1.172528f,  4.478233f,  0.811502f,  2.174423f,  1.307593f,  2.343714f,  1.130768f,  3.896447f,  0.0f,  2.938536f,  2.012876f,  14.086647f,  10.230303f,  0.808917f,  3.392321f,  2.016872f,  0.749815f,  0.0f,  1.147369f,  9.506415f,  0.0f,  1.708999f,  2.050454f,  1.332459f,  1.283639f,  0.0f,  1.354667f,  0.0f,  2.911175f,  4.318802f,  0.0f,  2.106789f,  1.506301f,  2.169779f,  5.826517f,  1.54878f,  0.749263f,  5.872832f,  0.0f,  0.985451f,  9.710891f,  2.326271f,  2.377997f,  1.364674f,  0.0f,  6.061156f,  1.446263f,  7.553903f,  2.373832f,  2.067141f,  5.877072f,  2.878187f,  4.025752f,  0.0f,  3.580612f,  0.0f,  3.476471f,  2.289575f,  25.439737f,  0.783951f,  5.227004f,  1.327237f,  0.0f,  1.537065f,  1.523238f,  0.892011f,  1.122027f,  79.588895f,  0.888889f,  1.540561f,  6.281283f,  0.0f,  6.209328f,  2.294747f,  0.975048f,  1.578866f,  0.0f,  2.898716f,  4.318802f,  1.410132f,  3.759476f,  10.80134f,  0.782743f,  2.008896f,  1.927894f,  13.423613f,  0.0f,  2.328938f,  6.762045f,  1.576416f,  1.544073f,  2.596802f,  5.548129f,  23.559398f,  2.184946f,  0.0f,  0.0f,  4.25773f,  25.721519f,  5.24386f,  0.849498f,  1.69616f,  2.347774f,  0.0f,  5.855908f,  5.586248f,  1.968992f,  3.648115f,  0.0f,  2.386375f,  2.988235f,  5.163903f,  1.985343f,  6.745237f,  2.966423f,  1013.033837f,  100.500922f,  8.310811f,  0.0f,  4.914138f,  0.0f,  0.0f,  1.893756f,  3.877863f,  1.586261f,  8.172932f,  3.227955f,  4.864149f,  5.326332f,  0.835526f,  1.776224f,  1.87627f,  0.0f,  1.035678f,  0.0f,  3.88156f,  2.434989f,  9.579267f,  61.28545f,  13.099117f,  0.580571f,  4.657884f,  0.976923f,  5.039063f,  0.724163f,  36.448378f,  10.10318f,  1.630819f,  0.593112f,  2.541586f,  0.0f,  9.220672f,  1.155197f,  1.417015f,  11.065995f,  0.0f,  0.0f,  1.10135f,  0.949533f,  0.866155f,  1.254321f,  1.87627f,  0.803797f,  1.474601f,  1.011952f,  2.401891f,  2.190834f,  2.753388f,  0.596945f,  2.514851f,  2.330275f,  5.847469f,  1.54878f,  0.879654f,  0.0f,  9.391129f,  0.0f,  7.9375f,  1.381373f,  2.861972f,  3.552448f,  2.297341f,  0.972249f,  1.127636f,  0.589327f,  1.473532f,  0.971319f,  0.621407f,  6.560121f,  2.10352f,  6.59206f,  2.567275f,  0.0f,  1.305913f,  3.742173f,  3.972623f,  1.992157f,  0.0f,  1.002962f,  9.948612f,  11.831122f,  11.073521f,  2.687831f,  1.205929f,  5.736069f,  0.660598f,  1.021106f,  1.66421f,  0.0f,  1.935238f,  3.749077f,  2.493252f,  13.501661f,  3.28006f,  3.2512f,  1.099567f,  1.720576f,  1.984375f,  1.806222f,  1.266037f,  1.608868f,  13.905906f,  0.697802f,  3.783985f,  5.963309f,  1.69616f,  1.005941f,  1.069474f,  7.596239f,  0.824675f,  0.908766f,  0.0f,  1.837251f,  1.348374f,  5.668049f,  1.28689f,  1.929725f,  2.332951f,  0.582569f,  1.458722f,  1.455587f,  1.383254f,  1.412092f,  1.918791f,  0.451957f,  3.320261f,  7.603379f,  1.322917f,  2.554366f,  0.0f,  1.697577f,  0.926162f,  2.271658f,  0.0f,  5.083172f,  0.0f,  1.471398f,  0.924477f,  0.565072f,  0.0f,  1.180703f,  0.632628f,  7.025075f,  1.807829f,  4.161806f,  1.011444f,  0.774981f,  57.902032f,  2.734859f,  2.194384f,  2.264065f,  1.080851f,  0.698282f,  0.770281f,  1.476744f,  1.495217f,  0.84106f,  0.374908f,  3.976517f,  4.589493f,  1.064427f,  0.0f,  1.094828f,  2.962099f,  1.402346f,  5.017284f,  1.195294f,  1.727891f,  8.353559f,  0.347945f,  2.783562f,  1.180023f,  0.945996f,  0.624462f,  2.290868f,  1.686307f,  0.951311f,  1.05558f,  4.728326f,  0.0f,  2.541586f,  1.212411f,  1.036735f,  0.578258f,  2.366917f,  1.483212f,  1.05176f,  3.470533f,  0.465201f,  0.600828f,  1.784021f,  0.46098f,  2.555975f,  83.04453f,  0.45236f,  0.915315f,  6.994817f,  0.930403f,  3.762963f,  0.354749f,  1.282828f,  0.0f,  1.086631f,  0.0f,  0.0f,  0.895944f,  0.985451f,  1.057232f,  0.99122f,  2.642393f,  1.063874f,  1.412092f,  2.810512f,  1.199528f,  8.05153f,  0.782743f,  0.866894f,  2.979472f,  1.588741f,  1.617834f,  0.472998f,  2.818308f,  1.078556f,  4.094714f,  3.863118f,  0.347232f,  2.323611f,  2.727517f,  0.761619f,  2.31699f,  2.129979f,  1.041513f,  0.238051f,  1.619124f,  0.0f,  3.392321f,  5.911266f,  1.438075f,  0.0f,  2.830084f,  11.909116f,  2.107884f,  2.021891f,  3.277419f,  8.892748f,  1.19389f,  0.861747f,  1.329843f,  1.486467f,  0.741065f,  2.366917f,  6.310559f,  2.150265f,  2.606797f,  7.491255f,  2.54f,  1.327237f,  0.563505f,  2.696746f,  2.533666f,  5.274503f,  1.684909f,  1.226307f,  0.0f,  1.791887f,  2.97293f,  1.10135f,  0.577929f,  0.0f,  1.54173f,  2.932179f,  0.938135f,  4.500546f,  1.248157f,  1.282019f,  0.885789f,  0.851635f,  1.325506f,  0.619135f,  5.017284f,  1.104944f,  13.603344f,  0.754269f,  2.585242f,  1.991179f,  0.0f,  103.955209f,  1.160474f,  1.458722f,  0.715493f,  1.178654f,  5.004926f,  7.980344f,  0.511581f,  1.115258f,  2.297341f,  1.131403f,  1.127636f,  0.829388f,  1.71912f, };

template<typename T>
void quantize_weights(cldnn::memory& weights, cldnn::memory& w_qf)
{
    auto batch_pitch = weights.get_layout().get_pitches().batch[0];
    auto ptr = weights.pointer<T>();
    auto wqf_ptr = w_qf.pointer<float>();
    T max = (T) 0.0f;
    for (int ofm = 0; ofm < weights.get_layout().size.batch[0]; ofm++)
    {
        max = (T) 0.0f;
        for (int w = 0; w < batch_pitch; w++)
            if (max < abs(ptr[ofm* batch_pitch + w]))
                max = abs(ptr[ofm* batch_pitch + w]);

        if (max == (T)0)
            max = (T)1; // do not quantize

        for (int w = 0; w < batch_pitch; w++)
            ptr[ofm* batch_pitch + w] = (T)round((float)ptr[ofm* batch_pitch + w] * 127.0f / (float)max);
        wqf_ptr[ofm] = max / 127.0f;
    }
}
template<typename T>
void calibrate(const cldnn::memory& output, cldnn::memory& calibrations)
{
    auto feature_pitch = output.get_layout().get_pitches().feature[0];
    auto ptr = output.pointer<T>();
    auto calibrations_ptr = calibrations.pointer<float>();
    T max = (T) 0.0f;
    for (int ofm = 0; ofm < output.get_layout().size.feature[0]; ofm++)
    {
        max = (T) 0.0f;
        for (int w = 0; w < feature_pitch; w++)
            if (max < abs(ptr[ofm* feature_pitch + w]))
                max = abs(ptr[ofm* feature_pitch + w]);
        calibrations_ptr[ofm] = 127.0f / max;
    }
}

template<typename T>
T max_abs(const cldnn::memory& mem)
{
    T max = (T)0;
    auto ptr = mem.pointer<T>();
    for (auto& a : ptr)
        if (max < abs(a))
            max = abs(a);
    return max;
}

template<typename T>
void apply_calibration_on_weights(cldnn::memory& weights,const std::vector<float>& qf)
{
    auto batch_pitch = weights.get_layout().get_pitches().batch[0];
    auto ptr = weights.pointer<T>();
    tensor w_size = weights.get_layout().size;
    int index = 0;
    for (int ofm = 0; ofm < w_size.batch[0]; ofm++)
        for (int ifm = 0; ifm < w_size.feature[0]; ifm++)
            for (int xy = 0; xy < w_size.spatial[0] * w_size.spatial[1]; xy++)
            {
                if (qf[ifm] != 0.0f)
                    ptr[index] = ptr[index] / qf[ifm];
                else
                    ptr[index] = (T)0;
                index++;
            }
}

cldnn::memory create_int8_weights(engine engine, cldnn::memory& in_weights)
{
    auto layout = in_weights.get_layout();
    auto out_weights = memory::allocate(engine, { data_types::i8, layout.format, layout.size });
    auto in = in_weights.pointer<float>();
    auto out = out_weights.pointer<char>();
    int indx = 0;
    for (auto& a : in)
        out[indx++] = (char)a;
    return out_weights;
}


topology build_squeezenet_quant(const std::string& weights_dir, const cldnn::engine& engine, cldnn::layout& input_layout, int32_t batch_size)
{
    // [227x227x3xB] convolution->relu->pooling->lrn [1000xB]
    input_layout.size = { batch_size, 3, 227, 227 };
    auto input = cldnn::input_layout("input", input_layout);

    //auto reorder_mean = { (float)104.0069879317889, (float)116.66876761696767, (float)122.6789143406786 };
    auto reordered_input = reorder(
        "reorder",
        input,
        { input_layout.data_type, input_layout.format, input_layout.size },
        std::vector<float>{ (float)104.0069879317889, (float)116.66876761696767, (float)122.6789143406786 });

    auto conv1_weights = file::create({ engine, join_path(weights_dir, "conv1_weights.nnd")});
    auto conv1_bias = file::create({ engine, join_path(weights_dir, "conv1_bias.nnd")});
    auto conv1 = convolution(
        "conv1",
        reordered_input,
        { conv1_weights },
        { conv1_bias },
        { 1,1,2,2 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto pool1 = pooling(
        "pool1",
        conv1,
        pooling_mode::max,
        { 1,1,3,3 }, // kernel
        { 1,1,2,2 }); // strd


    auto fire2_squeeze1x1_weights = file::create({ engine, join_path(weights_dir, "fire2_squeeze1x1_weights.nnd")});
    auto fire2_squeeze1x1_bias = file::create({ engine, join_path(weights_dir, "fire2_squeeze1x1_bias.nnd")});
    auto fire2_squeeze1x1 = convolution(
        "fire2_squeeze1x1",
        pool1,
        { fire2_squeeze1x1_weights },
        { fire2_squeeze1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire2_expand1x1_weights = file::create({ engine, join_path(weights_dir, "fire2_expand1x1_weights.nnd")});
    auto fire2_expand1x1_bias = file::create({ engine, join_path(weights_dir, "fire2_expand1x1_bias.nnd")});
    auto fire2_expand1x1 = convolution(
        "fire2_expand1x1",
        fire2_squeeze1x1,
        { fire2_expand1x1_weights },
        { fire2_expand1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire2_expand3x3_weights = file::create({ engine, join_path(weights_dir, "fire2_expand3x3_weights.nnd")});
    auto fire2_expand3x3_bias = file::create({ engine, join_path(weights_dir, "fire2_expand3x3_bias.nnd")});
    auto fire2_expand3x3 = convolution(
        "fire2_expand3x3",
        fire2_squeeze1x1,
        { fire2_expand3x3_weights },
        { fire2_expand3x3_bias },
        { 1,1,1,1 },
        { 0, 0, -1,-1 },
        { 1,1,1,1 },
        true);


    auto fire2_concat = concatenation(   
        "fire2_concat",
        {
            fire2_expand1x1,
            fire2_expand3x3
        },
        concatenation::along_f
    );


    auto fire3_squeeze1x1_weights = file::create({ engine, join_path(weights_dir, "fire3_squeeze1x1_weights.nnd")});
    auto fire3_squeeze1x1_bias = file::create({ engine, join_path(weights_dir, "fire3_squeeze1x1_bias.nnd")});
    auto fire3_squeeze1x1 = convolution(
        "fire3_squeeze1x1",
        fire2_concat,
        { fire3_squeeze1x1_weights },
        { fire3_squeeze1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire3_expand1x1_weights = file::create({ engine, join_path(weights_dir, "fire3_expand1x1_weights.nnd")});
    auto fire3_expand1x1_bias = file::create({ engine, join_path(weights_dir, "fire3_expand1x1_bias.nnd")});
    auto fire3_expand1x1 = convolution(
        "fire3_expand1x1",
        fire3_squeeze1x1,
        { fire3_expand1x1_weights },
        { fire3_expand1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire3_expand3x3_weights = file::create({ engine, join_path(weights_dir, "fire3_expand3x3_weights.nnd")});
    auto fire3_expand3x3_bias = file::create({ engine, join_path(weights_dir, "fire3_expand3x3_bias.nnd")});
    auto fire3_expand3x3 = convolution(
        "fire3_expand3x3",
        fire3_squeeze1x1,
        { fire3_expand3x3_weights },
        { fire3_expand3x3_bias },
        { 1,1,1,1 },
        { 0, 0, -1,-1 },
        { 1,1,1,1 },
        true);

    auto fire3_concat = concatenation(
        "fire3_concat",
        {
            fire3_expand1x1,
            fire3_expand3x3
        },
        concatenation::along_f
    );

    auto pool3 = pooling(
        "pool3",
        fire3_concat,
        pooling_mode::max,
        { 1,1,3,3 }, // kernel
        { 1,1,2,2 }); // strd

    auto fire4_squeeze1x1_weights = file::create({ engine, join_path(weights_dir, "fire4_squeeze1x1_weights.nnd")});
    auto fire4_squeeze1x1_bias = file::create({ engine, join_path(weights_dir, "fire4_squeeze1x1_bias.nnd")});
    auto fire4_squeeze1x1 = convolution(
        "fire4_squeeze1x1",
        pool3,
        { fire4_squeeze1x1_weights },
        { fire4_squeeze1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire4_expand1x1_weights = file::create({ engine, join_path(weights_dir, "fire4_expand1x1_weights.nnd")});
    auto fire4_expand1x1_bias = file::create({ engine, join_path(weights_dir, "fire4_expand1x1_bias.nnd")});
    auto fire4_expand1x1 = convolution(
        "fire4_expand1x1",
        fire4_squeeze1x1,
        { fire4_expand1x1_weights },
        { fire4_expand1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire4_expand3x3_weights = file::create({ engine, join_path(weights_dir, "fire4_expand3x3_weights.nnd")});
    auto fire4_expand3x3_bias = file::create({ engine, join_path(weights_dir, "fire4_expand3x3_bias.nnd")});
    auto fire4_expand3x3 = convolution(
        "fire4_expand3x3",
        fire4_squeeze1x1,
        { fire4_expand3x3_weights },
        { fire4_expand3x3_bias },
        { 1,1,1,1 },
        { 0, 0, -1,-1 },
        { 1,1,1,1 },
        true);

    auto fire4_concat = concatenation(
        "fire4_concat",
        {
            fire4_expand1x1,
            fire4_expand3x3
        },
        concatenation::along_f
    );

    auto fire5_squeeze1x1_weights = file::create({ engine, join_path(weights_dir, "fire5_squeeze1x1_weights.nnd")});
    auto fire5_squeeze1x1_bias = file::create({ engine, join_path(weights_dir, "fire5_squeeze1x1_bias.nnd")});
    auto fire5_squeeze1x1 = convolution(
        "fire5_squeeze1x1",
        fire4_concat,
        { fire5_squeeze1x1_weights },
        { fire5_squeeze1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire5_expand1x1_weights = file::create({ engine, join_path(weights_dir, "fire5_expand1x1_weights.nnd")});
    auto fire5_expand1x1_bias = file::create({ engine, join_path(weights_dir, "fire5_expand1x1_bias.nnd")});
    auto fire5_expand1x1 = convolution(
        "fire5_expand1x1",
        fire5_squeeze1x1,
        { fire5_expand1x1_weights },
        { fire5_expand1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire5_expand3x3_weights = file::create({ engine, join_path(weights_dir, "fire5_expand3x3_weights.nnd")});
    auto fire5_expand3x3_bias = file::create({ engine, join_path(weights_dir, "fire5_expand3x3_bias.nnd")});
    auto fire5_expand3x3 = convolution(
        "fire5_expand3x3",
        fire5_squeeze1x1,
        { fire5_expand3x3_weights },
        { fire5_expand3x3_bias },
        { 1,1,1,1 },
        { 0, 0, -1,-1 },
        { 1,1,1,1 },
        true);

    auto fire5_concat = concatenation(
        "fire5_concat",
        {
            fire5_expand1x1,
            fire5_expand3x3
        },
        concatenation::along_f
    );

    auto pool5 = pooling(
        "pool5",
        fire5_concat,
        pooling_mode::max,
        { 1,1,3,3 }, // kernel
        { 1,1,2,2 }); // strd

    auto fire6_squeeze1x1_weights = file::create({ engine, join_path(weights_dir, "fire6_squeeze1x1_weights.nnd")});
    auto fire6_squeeze1x1_bias = file::create({ engine, join_path(weights_dir, "fire6_squeeze1x1_bias.nnd")});
    auto fire6_squeeze1x1 = convolution(
        "fire6_squeeze1x1",
        pool5,
        { fire6_squeeze1x1_weights },
        { fire6_squeeze1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire6_expand1x1_weights = file::create({ engine, join_path(weights_dir, "fire6_expand1x1_weights.nnd")});
    auto fire6_expand1x1_bias = file::create({ engine, join_path(weights_dir, "fire6_expand1x1_bias.nnd")});
    auto fire6_expand1x1 = convolution(
        "fire6_expand1x1",
        fire6_squeeze1x1,
        { fire6_expand1x1_weights },
        { fire6_expand1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire6_expand3x3_weights = file::create({ engine, join_path(weights_dir, "fire6_expand3x3_weights.nnd")});
    auto fire6_expand3x3_bias = file::create({ engine, join_path(weights_dir, "fire6_expand3x3_bias.nnd")});
    auto fire6_expand3x3 = convolution(
        "fire6_expand3x3",
        fire6_squeeze1x1,
        { fire6_expand3x3_weights },
        { fire6_expand3x3_bias },
        { 1,1,1,1 },
        { 0, 0, -1,-1 },
        { 1,1,1,1 },
        true);

    auto fire6_concat = concatenation( 
        "fire6_concat",
        {
            fire6_expand1x1,
            fire6_expand3x3
        },
        concatenation::along_f
    );

    auto fire7_squeeze1x1_weights = file::create({ engine, join_path(weights_dir, "fire7_squeeze1x1_weights.nnd")});
    auto fire7_squeeze1x1_bias = file::create({ engine, join_path(weights_dir, "fire7_squeeze1x1_bias.nnd")});
    auto fire7_squeeze1x1 = convolution(
        "fire7_squeeze1x1",
        fire6_concat,
        { fire7_squeeze1x1_weights },
        { fire7_squeeze1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire7_expand1x1_weights = file::create({ engine, join_path(weights_dir, "fire7_expand1x1_weights.nnd")});
    auto fire7_expand1x1_bias = file::create({ engine, join_path(weights_dir, "fire7_expand1x1_bias.nnd")});
    auto fire7_expand1x1 = convolution(
        "fire7_expand1x1",
        fire7_squeeze1x1,
        { fire7_expand1x1_weights },
        { fire7_expand1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire7_expand3x3_weights = file::create({ engine, join_path(weights_dir, "fire7_expand3x3_weights.nnd")});
    auto fire7_expand3x3_bias = file::create({ engine, join_path(weights_dir, "fire7_expand3x3_bias.nnd")});
    auto fire7_expand3x3 = convolution(
        "fire7_expand3x3",
        fire7_squeeze1x1,
        { fire7_expand3x3_weights },
        { fire7_expand3x3_bias },
        { 1,1,1,1 },
        { 0, 0, -1,-1 },
        { 1,1,1,1 },
        true);

    auto fire7_concat = concatenation(
        "fire7_concat",
        {
            fire7_expand1x1,
            fire7_expand3x3
        },
        concatenation::along_f
    );

    auto fire8_squeeze1x1_weights = file::create({ engine, join_path(weights_dir, "fire8_squeeze1x1_weights.nnd")});
    auto fire8_squeeze1x1_bias = file::create({ engine, join_path(weights_dir, "fire8_squeeze1x1_bias.nnd")});
    auto fire8_squeeze1x1 = convolution(
        "fire8_squeeze1x1",
        fire7_concat,
        { fire8_squeeze1x1_weights },
        { fire8_squeeze1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire8_expand1x1_weights = file::create({ engine, join_path(weights_dir, "fire8_expand1x1_weights.nnd")});
    auto fire8_expand1x1_bias = file::create({ engine, join_path(weights_dir, "fire8_expand1x1_bias.nnd")});
    auto fire8_expand1x1 = convolution(
        "fire8_expand1x1",
        fire8_squeeze1x1,
        { fire8_expand1x1_weights },
        { fire8_expand1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire8_expand3x3_weights = file::create({ engine, join_path(weights_dir, "fire8_expand3x3_weights.nnd")});
    auto fire8_expand3x3_bias = file::create({ engine, join_path(weights_dir, "fire8_expand3x3_bias.nnd")});
    auto fire8_expand3x3 = convolution(
        "fire8_expand3x3",
        fire8_squeeze1x1,
        { fire8_expand3x3_weights },
        { fire8_expand3x3_bias },
        { 1,1,1,1 },
        { 0, 0, -1,-1 },
        { 1,1,1,1 },
        true);

    auto fire8_concat = concatenation(
        "fire8_concat",
        {
            fire8_expand1x1,
            fire8_expand3x3
        },
        concatenation::along_f
    );

    auto fire9_squeeze1x1_weights = file::create({ engine, join_path(weights_dir, "fire9_squeeze1x1_weights.nnd")});
    auto fire9_squeeze1x1_bias = file::create({ engine, join_path(weights_dir, "fire9_squeeze1x1_bias.nnd")});
    auto fire9_squeeze1x1 = convolution(
        "fire9_squeeze1x1",
        fire8_concat,
        { fire9_squeeze1x1_weights },
        { fire9_squeeze1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire9_expand1x1_weights = file::create({ engine, join_path(weights_dir, "fire9_expand1x1_weights.nnd")});
    auto fire9_expand1x1_bias = file::create({ engine, join_path(weights_dir, "fire9_expand1x1_bias.nnd")});
    auto fire9_expand1x1 = convolution(
        "fire9_expand1x1",
        fire9_squeeze1x1,
        { fire9_expand1x1_weights },
        { fire9_expand1x1_bias },
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto fire9_expand3x3_weights = file::create({ engine, join_path(weights_dir, "fire9_expand3x3_weights.nnd")});
    auto fire9_expand3x3_bias = file::create({ engine, join_path(weights_dir, "fire9_expand3x3_bias.nnd")});
    auto fire9_expand3x3 = convolution(
        "fire9_expand3x3",
        fire9_squeeze1x1,
        { fire9_expand3x3_weights },
        { fire9_expand3x3_bias },
        { 1,1,1,1 },
        { 0, 0, -1,-1 },
        { 1,1,1,1 },
        true);

    auto fire9_concat = concatenation(
        "fire9_concat",
        {
            fire9_expand1x1,
            fire9_expand3x3
        },
        concatenation::along_f
    );

    auto conv10_weights = file::create({ engine, join_path(weights_dir, "conv10_weights.nnd")});
    auto conv10_bias = file::create({ engine, join_path(weights_dir, "conv10_bias.nnd") });
                                                                   
    auto& conv10_w = conv10_weights.mem;
    apply_calibration_on_weights<float>(conv10_w, fire9_concat_calib);
    auto conv10_w_qf = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, conv10_w.get_layout().size.batch[0], 1, 1 } });
    auto conv10_o_qf = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, (int32_t)conv10_calib.size(), 1, 1 } });
    set_values(conv10_o_qf, conv10_calib);
    quantize_weights<float>(conv10_w, conv10_w_qf);
    auto conv10_w_int = create_int8_weights(engine, conv10_w);
    auto conv10_weigths_int = data("conv10_weights_int", conv10_w_int);
    auto conv10_weights_qf = data("conv10_w_qf", conv10_w_qf);
    auto conv10_output_qf = data("conv10_o_qf", conv10_o_qf);

    auto conv10_calibrator = reorder("conv10_calib", "fire9_concat",
        format::bfyx, data_types::i8, fire9_concat_calib, cldnn_reorder_mean_mode::mean_mul);

    auto conv10 = convolution(
        "conv10",
        conv10_calibrator,
        { conv10_weigths_int },
        { conv10_bias },
        { conv10_weights_qf },
        { conv10_output_qf },
        1.0f, // do not scale input
        { 1,1,1,1 },
        { 0,0,0,0 },
        { 1,1,1,1 },
        true);

    auto conv10_decalibrator = reorder("conv10_decalib", "conv10",
        format::bfyx, data_types::f32, conv10_calib, cldnn_reorder_mean_mode::mean_div);

    auto pool10 = pooling(
        "pool10",
        conv10_decalibrator,
        pooling_mode::average,
        { 1,1,14,14 }, // kernel
        { 1,1,1,1 }); // strd

    auto softmax = cldnn::softmax(
        "output",
        pool10);

    cldnn::topology topology(
        input,
        reordered_input,
        conv1, conv1_weights, conv1_bias,
        pool1,
        fire2_squeeze1x1, fire2_squeeze1x1_weights, fire2_squeeze1x1_bias
    );
    
    topology.add(fire2_expand1x1, fire2_expand1x1_weights, fire2_expand1x1_bias);
    topology.add(fire2_expand3x3, fire2_expand3x3_weights, fire2_expand3x3_bias);
        
    topology.add(
        fire2_concat);
    topology.add(
        fire3_squeeze1x1, fire3_squeeze1x1_weights, fire3_squeeze1x1_bias,
        fire3_expand1x1, fire3_expand1x1_weights, fire3_expand1x1_bias,
        fire3_expand3x3, fire3_expand3x3_weights, fire3_expand3x3_bias,
        fire3_concat,
        pool3);
    topology.add(
        fire4_squeeze1x1, fire4_squeeze1x1_weights, fire4_squeeze1x1_bias,
        fire4_expand1x1, fire4_expand1x1_weights, fire4_expand1x1_bias,
        fire4_expand3x3, fire4_expand3x3_weights, fire4_expand3x3_bias,
        fire4_concat);
    topology.add(
        fire5_squeeze1x1, fire5_squeeze1x1_weights, fire5_squeeze1x1_bias,
        fire5_expand1x1, fire5_expand1x1_weights, fire5_expand1x1_bias,
        fire5_expand3x3, fire5_expand3x3_weights, fire5_expand3x3_bias,
        fire5_concat,
        pool5);
    topology.add(
        fire6_squeeze1x1, fire6_squeeze1x1_weights, fire6_squeeze1x1_bias,
        fire6_expand1x1, fire6_expand1x1_weights, fire6_expand1x1_bias,
        fire6_expand3x3, fire6_expand3x3_weights, fire6_expand3x3_bias,
        fire6_concat);
    topology.add(
        fire7_squeeze1x1, fire7_squeeze1x1_weights, fire7_squeeze1x1_bias,
        fire7_expand1x1, fire7_expand1x1_weights, fire7_expand1x1_bias,
        fire7_expand3x3, fire7_expand3x3_weights, fire7_expand3x3_bias,
        fire7_concat);
    topology.add(
        fire8_squeeze1x1, fire8_squeeze1x1_weights, fire8_squeeze1x1_bias,
        fire8_expand1x1, fire8_expand1x1_weights, fire8_expand1x1_bias,
        fire8_expand3x3, fire8_expand3x3_weights, fire8_expand3x3_bias,
        fire8_concat);
    topology.add(
        fire9_squeeze1x1, fire9_squeeze1x1_weights, fire9_squeeze1x1_bias,
        fire9_expand1x1, fire9_expand1x1_weights, fire9_expand1x1_bias,
        fire9_expand3x3, fire9_expand3x3_weights, fire9_expand3x3_bias,
        fire9_concat);
    topology.add(
        conv10, conv10_bias,
        pool10,
        softmax);
    topology.add(
        conv10_calibrator, conv10_weigths_int,
        conv10_weights_qf, conv10_output_qf,
        conv10_decalibrator);
    return topology;
}
