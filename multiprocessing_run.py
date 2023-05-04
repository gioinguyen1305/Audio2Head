import torch
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, Queue


class Multiprocessor:
    def __init__(self, generator, opt, img):
        self.processes = {}
        self.generator = generator
        self.opt = opt
        self.img = img
        self.queue = Queue(200)
        self.len_frame = 0

    def run_prediction(self, kp_gen_source_array, gen_kp_jacobian, gen_kp_value, bs_idx, startid, end_id, queue):
        gen_kp_jacobian = torch.from_numpy(gen_kp_jacobian)
        gen_kp_value = torch.from_numpy(gen_kp_value)
        kp_gen_source_tensor = {k:torch.from_numpy(v) for k, v in kp_gen_source_array.items()}

        for frame_bs_idx in tqdm(range(startid, end_id)):
            self.len_frame = self.len_frame + 1

            tt = {}
            tt["value"] = gen_kp_value[:, frame_bs_idx]

            if self.opt.estimate_jacobian:
                tt["jacobian"] = gen_kp_jacobian[:, frame_bs_idx]
            out_gen = self.generator(self.img, kp_source=kp_gen_source_tensor, kp_driving=tt)
            queue.put({str(bs_idx) + "_" + str(frame_bs_idx): (np.transpose(out_gen['prediction'].data.cpu().numpy(),
                                                                                [0, 2, 3, 1])[0] * 255).astype(np.uint8)})
            del out_gen, tt
        del kp_gen_source_tensor, gen_kp_value, gen_kp_jacobian
        queue.put({bs_idx: None})

    def run(self, kp_gen_source_array, gen_kp_jacobian, gen_kp_value, thread, startid, end_id):
        p = Process(target=self.run_prediction, args=(kp_gen_source_array, gen_kp_jacobian, gen_kp_value, thread, startid, end_id, self.queue, ))
        self.processes[thread] = p
        p.start()

    def wait(self):
        rets = {}
        while any([v.is_alive() for v in self.processes.values()]):
            try:
                output = self.queue.get(timeout=1)
            except:
                continue
            if list(output.values())[0] is None:
                self.processes[int(list(output.keys())[0])].terminate()
            else:
                key = int(list(output.keys())[0].split("_")[-1])
                thread_ = int(list(output.keys())[0].split("_")[0])
                if thread_ not in rets.keys():
                    rets[thread_] = {}
                rets[thread_][key] = list(output.values())[0]
        return rets
