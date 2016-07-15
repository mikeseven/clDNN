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

#include "FreeImage_wraps.h"

FIBITMAP* fi::load_image_from_file( std::string filename )
{
    FIBITMAP          *bitmap=nullptr;
    FREE_IMAGE_FORMAT  image_format= FIF_UNKNOWN;


    // Attempting to obtain information about the file format based on extension
    image_format = FreeImage_GetFIFFromFilename( filename.c_str() );

    if ( image_format != FIF_UNKNOWN ) {
        bitmap = FreeImage_Load( image_format,filename.c_str() );
        if ( bitmap != nullptr ) return bitmap;
    }
    // Attempting to open the file, in spite of the lack of information about the file format
    for ( FREE_IMAGE_FORMAT iff : fi::common_formats ) {
        bitmap = FreeImage_Load( iff,filename.c_str() );
        if ( bitmap != nullptr ) return bitmap;
    }

    return bitmap;
}

FIBITMAP * fi::crop_image_to_square_and_resize( FIBITMAP * bmp_in,uint16_t new_size )
{
    uint16_t  width,height;
    FIBITMAP *bmp_temp1 = nullptr;
    FIBITMAP *bmp_temp2 = nullptr;
    uint16_t  margin;

    width = FreeImage_GetWidth( bmp_in );
    height = FreeImage_GetHeight( bmp_in );

    if ( width!=height ) {
        if ( width>height ) {
            margin = (width - height)/2;
            bmp_temp1 = FreeImage_Copy( bmp_in,margin,0,width-margin-1,height );
        }
        else {
            margin = (height - width)/2;
            bmp_temp1 = FreeImage_Copy( bmp_in,0,margin,width,height-margin-1 );
        }
        FreeImage_Unload( bmp_in );
    }
    else
        bmp_temp1 = bmp_in;

    bmp_temp2 = FreeImage_Rescale( bmp_temp1,new_size,new_size, FILTER_CATMULLROM );

    FreeImage_Unload( bmp_temp1 );

    return bmp_temp2;
}

FIBITMAP * fi::resize_image_to_square(FIBITMAP * bmp_in, uint16_t new_size)
{
    FIBITMAP *bmp_temp1 = nullptr;
    bmp_temp1 = FreeImage_Rescale(bmp_in, new_size, new_size, FILTER_CATMULLROM);
    FreeImage_Unload(bmp_in);
    return bmp_temp1;
}
