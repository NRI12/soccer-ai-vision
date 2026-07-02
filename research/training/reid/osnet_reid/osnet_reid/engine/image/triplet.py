from __future__ import division, print_function, absolute_import

import torch
from osnet_reid import metrics
from osnet_reid.losses import TripletLoss, CrossEntropyLoss

from ..engine import Engine


class ImageTripletEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``osnet_reid.data.ImageDataManager``
            or ``osnet_reid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import osnet_reid
        datamanager = osnet_reid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = osnet_reid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = osnet_reid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = osnet_reid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = osnet_reid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        weight_tc=0,
        weight_cc=0,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        topk=1,
        bottomk=1,
        warmup_lr=0.0,
        warmup_steps=0,
        lr=0.0003,
        use_amp=False,
    ):
        super(ImageTripletEngine, self).__init__(datamanager, use_gpu, warmup_steps, use_amp)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)
        self.lr_use_warmup = warmup_steps > 0
        self.warmup_lr = warmup_lr
        self.warmup_steps = warmup_steps
        self.lr = lr

        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_tc = weight_tc
        self.weight_cc = weight_cc

        self.criterion_t = TripletLoss(margin=margin,
                                       num_instances=datamanager.num_instances,
                                       topk=topk,
                                       bottomk=bottomk)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        amp_enabled = self.scaler is not None

        # Only the model forward runs in fp16; losses need fp32 for numerical stability
        with torch.autocast('cuda', enabled=amp_enabled):
            outputs, features = self.model(imgs)

        if amp_enabled:
            features = features.float()
            outputs = outputs.float()

        loss = 0
        loss_summary = {}

        if self.weight_t > 0 or self.weight_tc > 0 or self.weight_cc > 0:
            loss_t, loss_tc, loss_cc = self.compute_loss(self.criterion_t, features, pids, epoch=self.epoch)
            loss += self.weight_t * loss_t
            loss += self.weight_tc * loss_tc
            loss += self.weight_cc * loss_cc
            loss_summary['loss_t'] = loss_t.item()

        if self.weight_x > 0:
            loss_x = self.compute_loss(self.criterion_x, outputs, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = loss_x.item()
            loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        assert loss_summary

        self.optimizer.zero_grad()
        if amp_enabled:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss_summary