import sys
import pickle
import numpy as np
sys.path.append("..")

from gan import output
sys.modules["output"] = output

from gan.output import OutputType, Output, Normalization
from gan.doppelganger import DoppelGANger
from gan.util import add_gen_flag, normalize_per_sample
from gan.load_data import load_data
from gan.network import DoppelGANgerGenerator, Discriminator, AttrDiscriminator
import os
import tensorflow as tf


# (data_feature, data_attribute, data_gen_flag, data_feature_outputs, data_attribute_outputs) = load_data("../data/web")
# print(data_feature.shape)
# print(data_attribute.shape)
# print(data_gen_flag.shape)

# +
sample_len = 7
path = "../data/energy"

data_npz = np.load(
    os.path.join(path, "data_train.npz"))

with open(os.path.join(path, "data_feature_output.pkl"), "rb") as f:
    data_feature_outputs = pickle.load(f)    

with open(os.path.join(path, "data_attribute_output.pkl"), "rb") as f:
    data_attribute_outputs = pickle.load(f)

data_feature = data_npz["arr_0"]
data_attribute = data_npz["arr_1"]
data_gen_flag = data_npz["arr_2"]

print(data_feature.shape) 
print(data_attribute.shape)
print(data_gen_flag.shape)
# -

# (data_feature, data_attribute, data_attribute_outputs,
#  real_attribute_mask) = \
#     normalize_per_sample(
#         data_feature, data_attribute, data_feature_outputs,
#         data_attribute_outputs)

# +
data_feature_min = np.amin(data_feature, axis=1)
data_feature_max = np.amax(data_feature, axis=1)

additional_attribute = []
additional_attribute_outputs = []

dim = 0
for output in data_feature_outputs:
    if output.type_ == OutputType.CONTINUOUS:
        for _ in range(output.dim):
            max_ = data_feature_max[:, dim]
            min_ = data_feature_min[:, dim]

            additional_attribute.append((max_ + min_) / 2.0)
            additional_attribute.append((max_ - min_) / 2.0)
            additional_attribute_outputs.append(Output(
                type_=OutputType.CONTINUOUS,
                dim=1,
                normalization=output.normalization,
                is_gen_flag=False))
            additional_attribute_outputs.append(Output(
                type_=OutputType.CONTINUOUS,
                dim=1,
                normalization=Normalization.ZERO_ONE,
                is_gen_flag=False))

            max_ = np.expand_dims(max_, axis=1)
            min_ = np.expand_dims(min_, axis=1)

            data_feature[:, :, dim] = \
                (data_feature[:, :, dim] - min_) / (max_ - min_)
            if output.normalization == Normalization.MINUSONE_ONE:
                data_feature[:, :, dim] = \
                    data_feature[:, :, dim] * 2.0 - 1.0

            dim += 1
    else:
        dim += output.dim

real_attribute_mask = ([True] * len(data_attribute_outputs) +
                       [False] * len(additional_attribute_outputs))

additional_attribute = np.stack(additional_attribute, axis=1)
data_attribute = np.concatenate([data_attribute, additional_attribute], axis=1)
data_attribute_outputs.extend(additional_attribute_outputs)

print(real_attribute_mask)
print(data_feature.shape)
print(data_attribute.shape)
print(len(data_attribute_outputs))
# -

sample_len = 7

# data_feature, data_feature_outputs = add_gen_flag(data_feature, data_gen_flag, data_feature_outputs, sample_len)

# +
for output in data_feature_outputs:
    if output.is_gen_flag:
        raise Exception("is_gen_flag should be False for all"
                        "feature_outputs")

if (data_feature.shape[2] !=
        np.sum([t.dim for t in data_feature_outputs])):
    raise Exception("feature dimension does not match feature_outputs")

if len(data_gen_flag.shape) != 2:
    raise Exception("data_gen_flag should be 2 dimension")

num_sample, length = data_gen_flag.shape

data_gen_flag = np.expand_dims(data_gen_flag, 2)

data_feature_outputs.append(Output(
    type_=OutputType.DISCRETE,
    dim=2,
    is_gen_flag=True))

shift_gen_flag = np.concatenate(
    [data_gen_flag[:, 1:, :],
     np.zeros((data_gen_flag.shape[0], 1, 1))],
    axis=1)

if length % sample_len != 0:
    raise Exception("length must be a multiple of sample_len")
data_gen_flag_t = np.reshape(
    data_gen_flag,
    [num_sample, int(length / sample_len), sample_len])
data_gen_flag_t = np.sum(data_gen_flag_t, 2)
data_gen_flag_t = data_gen_flag_t > 0.5
data_gen_flag_t = np.repeat(data_gen_flag_t, sample_len, axis=1)
data_gen_flag_t = np.expand_dims(data_gen_flag_t, 2)
data_feature = np.concatenate(
    [data_feature,
     shift_gen_flag,
     (1 - shift_gen_flag) * data_gen_flag_t],
    axis=2)

print(data_feature.shape)
print(len(data_feature_outputs))
# -

generator = DoppelGANgerGenerator(
    feed_back=False,
    noise=True,
    feature_outputs=data_feature_outputs,
    attribute_outputs=data_attribute_outputs,
    real_attribute_mask=real_attribute_mask,
    sample_len=sample_len)
discriminator = Discriminator()
attr_discriminator = AttrDiscriminator()

# +
checkpoint_dir = "./test/checkpoint"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
sample_dir = "./test/sample"
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
time_path = "./test/time.txt"
epoch = 400
batch_size = 100
vis_freq = 200
vis_num_sample = 5
d_rounds = 1
g_rounds = 1
d_gp_coe = 10.0
attr_d_gp_coe = 10.0
g_attr_d_coe = 1.0
extra_checkpoint_freq = 5
num_packing = 1

data_gen_flag = data_gen_flag.squeeze()
# -

run_config = tf.ConfigProto()
with tf.Session(config=run_config) as sess:
    gan = DoppelGANger(
        sess=sess,
        checkpoint_dir=checkpoint_dir,
        sample_dir=sample_dir,
        time_path=time_path,
        epoch=epoch,
        batch_size=batch_size,
        data_feature=data_feature,
        data_attribute=data_attribute,
        real_attribute_mask=real_attribute_mask,
        data_gen_flag=data_gen_flag,
        sample_len=sample_len,
        data_feature_outputs=data_feature_outputs,
        data_attribute_outputs=data_attribute_outputs,
        vis_freq=vis_freq,
        vis_num_sample=vis_num_sample,
        generator=generator,
        discriminator=discriminator,
        attr_discriminator=attr_discriminator,
        d_gp_coe=d_gp_coe,
        attr_d_gp_coe=attr_d_gp_coe,
        g_attr_d_coe=g_attr_d_coe,
        d_rounds=d_rounds,
        g_rounds=g_rounds,
        num_packing=num_packing,
        extra_checkpoint_freq=extra_checkpoint_freq)
    gan.build()
    gan.train()

# if __name__ == "__main__":
#     sample_len = 10
#
#     (data_feature, data_attribute,
#      data_gen_flag,
#      data_feature_outputs, data_attribute_outputs) = \
#         load_data("../data/web")
#     print(data_feature.shape)
#     print(data_attribute.shape)
#     print(data_gen_flag.shape)
#
#     (data_feature, data_attribute, data_attribute_outputs,
#      real_attribute_mask) = \
#         normalize_per_sample(
#             data_feature, data_attribute, data_feature_outputs,
#             data_attribute_outputs)
#     print(real_attribute_mask)
#     print(data_feature.shape)
#     print(data_attribute.shape)
#     print(len(data_attribute_outputs))
#
#     data_feature, data_feature_outputs = add_gen_flag(
#         data_feature, data_gen_flag, data_feature_outputs, sample_len)
#     print(data_feature.shape)
#     print(len(data_feature_outputs))
#
#     generator = DoppelGANgerGenerator(
#         feed_back=False,
#         noise=True,
#         feature_outputs=data_feature_outputs,
#         attribute_outputs=data_attribute_outputs,
#         real_attribute_mask=real_attribute_mask,
#         sample_len=sample_len)
#     discriminator = Discriminator()
#     attr_discriminator = AttrDiscriminator()
#
#     checkpoint_dir = "./test/checkpoint"
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#     sample_dir = "./test/sample"
#     if not os.path.exists(sample_dir):
#         os.makedirs(sample_dir)
#     time_path = "./test/time.txt"
#     epoch = 400
#     batch_size = 100
#     vis_freq = 200
#     vis_num_sample = 5
#     d_rounds = 1
#     g_rounds = 1
#     d_gp_coe = 10.0
#     attr_d_gp_coe = 10.0
#     g_attr_d_coe = 1.0
#     extra_checkpoint_freq = 5
#     num_packing = 1
#
#     run_config = tf.ConfigProto()
#     with tf.Session(config=run_config) as sess:
#         gan = DoppelGANger(
#             sess=sess,
#             checkpoint_dir=checkpoint_dir,
#             sample_dir=sample_dir,
#             time_path=time_path,
#             epoch=epoch,
#             batch_size=batch_size,
#             data_feature=data_feature,
#             data_attribute=data_attribute,
#             real_attribute_mask=real_attribute_mask,
#             data_gen_flag=data_gen_flag,
#             sample_len=sample_len,
#             data_feature_outputs=data_feature_outputs,
#             data_attribute_outputs=data_attribute_outputs,
#             vis_freq=vis_freq,
#             vis_num_sample=vis_num_sample,
#             generator=generator,
#             discriminator=discriminator,
#             attr_discriminator=attr_discriminator,
#             d_gp_coe=d_gp_coe,
#             attr_d_gp_coe=attr_d_gp_coe,
#             g_attr_d_coe=g_attr_d_coe,
#             d_rounds=d_rounds,
#             g_rounds=g_rounds,
#             num_packing=num_packing,
#             extra_checkpoint_freq=extra_checkpoint_freq)
#         gan.build()
#         gan.train()

#         '''
#         {
#             "dataset": ["google"],
#             "epoch": [400],
#             "run": [0, 1, 2],
#             "sample_len": [1, 5, 10],
#             "extra_checkpoint_freq": [5],
#             "epoch_checkpoint_freq": [1],
#             "aux_disc": [False],
#             "self_norm": [False]
#         },
#         {
#             "dataset": ["web"],
#             "epoch": [400],
#             "run": [0, 1, 2],
#             "sample_len": [1, 5, 10, 25, 50],
#             "extra_checkpoint_freq": [5],
#             "epoch_checkpoint_freq": [1],
#             "aux_disc": [True],
#             "self_norm": [True]
#         },
#         {
#             "dataset": ["FCC_MBA"],
#             "epoch": [17000],
#             "run": [0, 1, 2],
#             "sample_len": [1, 4, 8],
#             "extra_checkpoint_freq": [850],
#             "epoch_checkpoint_freq": [70],
#             "aux_disc": [False],
#             "self_norm": [False]
#         }
#         '''
