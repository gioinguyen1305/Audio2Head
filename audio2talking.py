import subprocess
import torch
import cv2
from tqdm import tqdm
from skimage import io, img_as_float32
import numpy as np
from modules.audio2pose import get_pose_from_audio
from multiprocessing_run import Multiprocessor
from util import *


class A2T:
    def __init__(self):
        self.thread_num = 2
        self.model_path = r"./checkpoints/audio2head.pth.tar"
        self.img_path = r"./demo/img/paint.jpg"
        self.save_path = r"./results"
        self.temp_audio = "./results/temp.wav"
        self.config_file = r"./config/vox-256.yaml"
        self.config, self.opt = load_config_file(self.config_file)
        self.kp_detector, self.generator, self.audio2kp, self.audio2pose = load_model(self.config,
                                                                                      self.opt,
                                                                                      self.model_path)
        self.read_image()
        self.mp = Multiprocessor(self.generator, self.opt, self.img)

    def read_image(self):
        img = io.imread(self.img_path)[:, :, :3]
        img = cv2.resize(img, (256, 256))
        img = np.array(img_as_float32(img))
        img = img.transpose((2, 0, 1))
        self.img = torch.from_numpy(img).unsqueeze(0)
        kp_gen_source = self.kp_detector(self.img)
        self.kp_gen_source_array = {k: v.detach().numpy() for k, v in kp_gen_source.items()}
        del img, kp_gen_source

    def async_16000(self, audio_path):
        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (audio_path, self.temp_audio))
        subprocess.call(command, shell=True, stdout=None)

    def create_pose_with_audio(self):
        audio_f = []
        poses = []
        pad = np.zeros((4, 41), dtype=np.float32)
        for i in range(0, self.frames, self.opt.seq_len // 2):
            temp_audio = []
            temp_pos = []
            for j in range(self.opt.seq_len):
                if i + j < self.frames:
                    temp_audio.append(self.audio_feature[(i + j) * 4:(i + j) * 4 + 4])
                    trans = self.ref_pose_trans[i + j]
                    rot = self.ref_pose_rot[i + j]
                else:
                    temp_audio.append(pad)
                    trans = self.ref_pose_trans[-1]
                    rot = self.ref_pose_rot[-1]

                pose = np.zeros([256, 256])
                draw_annotation_box(pose, np.array(rot), np.array(trans))
                temp_pos.append(pose)
            audio_f.append(temp_audio)
            poses.append(temp_pos)
            del temp_pos, temp_audio, pose
        audio_f = torch.from_numpy(np.array(audio_f, dtype=np.float32)).unsqueeze(0)
        poses = torch.from_numpy(np.array(poses, dtype=np.float32)).unsqueeze(0)
        return audio_f, poses

    def run(self, audio_path):
        self.async_16000(audio_path)
        self.audio_feature = get_audio_feature_from_audio(self.temp_audio)
        self.frames = len(self.audio_feature) // 4

        self.ref_pose_rot, self.ref_pose_trans = get_pose_from_audio(self.img,
                                                                     self.audio_feature,
                                                                     self.audio2pose)
        audio_f, poses = self.create_pose_with_audio()
        bs = audio_f.shape[1]
        predictions_gen = []
        total_frames = 0
        for bs_idx in tqdm(range(0, bs, self.thread_num)):
            for td in range(self.thread_num):
                if td + bs_idx >= bs:
                    continue
                if td + bs_idx == 0:
                    startid = 0
                    end_id = self.opt.seq_len // 4 * 3
                else:
                    startid = self.opt.seq_len // 4
                    end_id = self.opt.seq_len // 4 * 3
                if bs_idx + td != 0:
                    if end_id - startid + total_frames > self.frames:
                        end_id = self.frames - total_frames + startid
                total_frames = total_frames + (end_id - startid)

                gen_kp = self.audio2kp({"audio": audio_f[:, td + bs_idx],
                                   "pose": poses[:, td + bs_idx],
                                   "id_img": self.img})
                gen_kp_value = gen_kp["value"].detach().numpy()
                gen_kp_jacobian = gen_kp["jacobian"].detach().numpy()
                self.mp.run(self.kp_gen_source_array, gen_kp_jacobian, gen_kp_value, td + bs_idx, startid, end_id)
            ret = self.mp.wait()
            for i in sorted(list(ret.keys()), reverse=False):
                predictions_gen = predictions_gen + list(dict(sorted(ret[i].items())).values())
            del gen_kp, gen_kp_value, gen_kp_jacobian, ret

        save_video(self.save_path, self.img_path, audio_path, predictions_gen)
