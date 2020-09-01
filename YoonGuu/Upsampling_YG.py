"""
not finished coding by YG
"""
# main imports
import sys
import os
# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf

dir_VoxelMorph = os.path.join(os.getcwd(), 'VoxelMorph')
dir_voxelmorph = os.path.join(dir_VoxelMorph, 'voxelmorph') #this dir_voxelmorph is diff from dir_VoxelMorph

# import neuron layers, which will be useful for Transforming.
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
sys.path.append(os.path.join(dir_VoxelMorph, 'YoonGuu'))

import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from keras.layers import LeakyReLU, Reshape, Lambda


def en_decom_size_ck_2D(medical_img_dim, enc_nf=[32, 32, 32, 32], dec_nf= [32, 32, 32, 32, 32, 16], strides=2):
    '''
    maden by YG

    :param medical_img_dim: image dimensions that will pass the encoding and decoding
    :param enc_nf:  voxelmorph's enc_nf
    :param dec_nf: voxelmorph's dec_nf
    :param strides:  voxelmorph's strides
    :return: correct dimension size to apply unet architecture which applies encoding and decoding
    '''
    # x_enc_list=x_enc
    # enc_nf = [32, 32, 32, 32]
    # dec_nf = [32, 32, 32, 32, 32, 16]
    # nf= 32
    # strides=2


    # medical_img_dim= dim_MRA_xy.shape
    # medical_img_dim = (182,182)
    # (218, 182)

    print('current dimension : ', medical_img_dim)
    #check the right dimension
    dim_holder=[]

    for i in medical_img_dim:
        print('medical_img_dim:', i)
        if not i in [16*x for x in range(30)]:
            for j in [16*y for y in range(30)]:
                if i<j:
                    dim_holder.append(j)
                    break
        else:
            dim_holder.append(i)

    medical_img_dim=tuple(dim_holder)

    print('dimension size should be change into ', medical_img_dim )

    #shows encoded, decoded dimension size
    src_feats = 1
    tgt_feats = 1
    src = None
    tgt = None
    ndims = len(medical_img_dim)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # inputs
    if src is None:
        src = Input(shape=[*medical_img_dim, src_feats])
    if tgt is None:
        tgt = Input(shape=[*medical_img_dim, tgt_feats])
    x_in = concatenate([src, tgt])

    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        # i =0
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    encode_dim_list = []
    for enc in x_enc:
        encode_dim_list.append(enc.get_shape().as_list())

    decode_dim_list=[]
    decode_dim_list.append(x_enc[-1].get_shape().as_list())
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    x_encode_expected = upsample_layer()(x_enc[-1])
    decode_dim_list.append(x_encode_expected.get_shape().as_list())
    x_encode_expected = upsample_layer()(x_encode_expected)
    decode_dim_list.append(x_encode_expected.get_shape().as_list())
    x_encode_expected = upsample_layer()(x_encode_expected)
    decode_dim_list.append(x_encode_expected.get_shape().as_list())
    x_encode_expected = upsample_layer()(x_encode_expected)
    decode_dim_list.append(x_encode_expected.get_shape().as_list())
    decode_dim_list.reverse()

    print('encode_dim_list : ', encode_dim_list)
    print('decode_dim_list : ', decode_dim_list)
    return medical_img_dim

def conv_block(x_in, nf, strides=1):
    #conv_block(x_enc[-1], enc_nf[i], 2)
    #x_in =x_enc[-1]
    #nf=enc_nf[i]
    #strides=2
    """
    specific convolution module including convolution followed by leakyrelu
    """

    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

