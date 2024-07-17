# encoding: utf-8
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 2
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1
        
        self.data_dir = "./datasets/two_classes"
        self.train_ann = "train.json"
        self.val_ann = "valid.json"

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.output_dir = '/mnt/data/nissay/task-2/models/yolox'

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
                cache=cache,
                cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="valid",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
