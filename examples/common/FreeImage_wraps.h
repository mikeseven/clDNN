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


#pragma once
#include <FreeImage.h>
#include <stdint.h>
#include <string>
namespace fi {

#ifdef __linux__
    const FREE_IMAGE_FORMAT common_formats[] = { FIF_JPEG, FIF_J2K, FIF_JP2, FIF_PNG, FIF_BMP, FIF_GIF, FIF_TIFF };
#else
    const FREE_IMAGE_FORMAT common_formats[] = { FIF_JPEG, FIF_J2K, FIF_JP2, FIF_PNG, FIF_BMP, FIF_WEBP, FIF_GIF, FIF_TIFF };
#endif

    FIBITMAP * load_image_from_file( std::string );
    FIBITMAP * crop_image_to_square_and_resize( FIBITMAP *,uint16_t );
    FIBITMAP * resize_image_to_square(FIBITMAP *, uint16_t);

    typedef  FIBITMAP* (*prepare_image_t)(FIBITMAP *, uint16_t);

}
