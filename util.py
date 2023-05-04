import pyworld
import torch
import cv2
import argparse
import python_speech_features
import numpy as np
import yaml, os, imageio
from scipy.interpolate import interp1d
from scipy.io import wavfile
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.audio2kp import AudioModel3D
from modules.audio2pose import audio2poseLSTM


def load_config_file(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    opt = argparse.Namespace(**yaml.load(open("./config/parameters.yaml"), Loader=yaml.Loader))
    return config, opt


def load_model(config, opt, model_path):
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    audio2kp = AudioModel3D(opt)
    audio2pose = audio2poseLSTM()

    checkpoint = torch.load(model_path, map_location="cpu")
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    generator.load_state_dict(checkpoint["generator"])
    audio2kp.load_state_dict(checkpoint["audio2kp"])
    audio2pose.load_state_dict(checkpoint["audio2pose"])

    audio2pose.eval()
    generator.eval()
    kp_detector.eval()
    audio2kp.eval()
    del checkpoint

    return kp_detector, generator, audio2kp, audio2pose


def inter_pitch(y, y_flag):
    frame_num = y.shape[0]
    i = 0
    last = -1
    while (i < frame_num):
        if y_flag[i] == 0:
            while True:
                if y_flag[i] == 0:
                    if i == frame_num - 1:
                        if last != -1:
                            y[last + 1:] = y[last]
                        i += 1
                        break
                    i += 1
                else:
                    break
            if i >= frame_num:
                break
            elif last == -1:
                y[:i] = y[i]
            else:
                inter_num = i - last + 1
                fy = np.array([y[last], y[i]])
                fx = np.linspace(0, 1, num=2)
                f = interp1d(fx, fy)
                fx_new = np.linspace(0, 1, inter_num)
                fy_new = f(fx_new)
                y[last + 1:i] = fy_new[1:-1]
                last = i
                i += 1

        else:
            last = i
            i += 1
    return y


def get_audio_feature_from_audio(audio_path, norm=True):
    sample_rate, audio = wavfile.read(audio_path)
    if len(audio.shape) == 2:
        if np.min(audio[:, 0]) <= 0:
            audio = audio[:, 1]
        else:
            audio = audio[:, 0]
    if norm:
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        a = python_speech_features.mfcc(audio, sample_rate)
        b = python_speech_features.logfbank(audio, sample_rate)
        c, _ = pyworld.harvest(audio, sample_rate, frame_period=10)
        c_flag = (c == 0.0) ^ 1
        c = inter_pitch(c, c_flag)
        c = np.expand_dims(c, axis=1)
        c_flag = np.expand_dims(c_flag, axis=1)
        frame_num = np.min([a.shape[0], b.shape[0], c.shape[0]])

        cat = np.concatenate([a[:frame_num], b[:frame_num], c[:frame_num], c_flag[:frame_num]], axis=1)
        return cat

def draw_annotation_box( image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""

    camera_matrix = np.array(
        [[233.333, 0, 128],
         [0, 233.333, 128],
         [0, 0, 1]], dtype="double")

    dist_coeefs = np.zeros((4, 1))

    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


def save_video(save_path, img_path, audio_path, predictions_gen):
    if not os.path.exists(os.path.join(save_path, "temp")):
        os.makedirs(os.path.join(save_path, "temp"))
    image_name = os.path.basename(img_path)[:-4] + "_" + os.path.basename(audio_path)[:-4] + ".mp4"

    video_path = os.path.join(save_path, "temp", image_name)

    imageio.mimsave(video_path, predictions_gen, fps=30.0)

    save_video = os.path.join(save_path, image_name)
    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (video_path, audio_path, save_video)
    os.system(cmd)
    os.remove(video_path)
