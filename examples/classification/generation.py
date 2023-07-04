import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from examples.hrank_utils.feature_extractor import RankExtractor


def get_feature_hook(self, input, output):
    global feature_result
    global entropy
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def get_feature_hook_densenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b - 12, b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def get_feature_hook_googlenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b - 12, b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def get_features_by_keys(input_features_dim, data):
    if input_features_dim == 3:
        features = data['pos']
    elif input_features_dim == 4:
        features = torch.cat(
            (data['pos'], data['heights']), dim=-1)
        raise NotImplementedError("error")
    return features.transpose(1, 2).contiguous()


def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
             cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)


def main(gpu, cfg, profile=False):
    cof_factor = cfg.cof_factor
    dis_factor = cfg.dis_factor
    rank_factor = cfg.rank_factor
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    num_points = val_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points sampled from dataset: {num_points}, "
                 f"number of points as model input: {cfg.num_points}")
    cfg.classes = cfg.get('classes', None) or val_loader.dataset.classes if hasattr(
        val_loader.dataset, 'classes') else None or np.range(num_classes)
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    epoch, best_val = load_checkpoint(model, cfg.pretrained_path)

    # feat_ext = RankExtractor(model, layers=layers_id)

    # for i in range(1):
    #     test(feat_ext.model, val_loader, validate_fn, epoch, cfg)
    for i in range(1):
        test(model, val_loader, validate_fn, epoch, cfg)
    hrank_channels_norm = []
    dis_channels_norm = []
    cof_channels_norm = []
    for child_module in model.modules():
        if hasattr(child_module, 'cof_channels_norm'):
            tmp_module_channel_norm = child_module.cof_channels_norm
            for i in range(len(tmp_module_channel_norm)):
                cof_channels_norm.append(tmp_module_channel_norm[i])
        if hasattr(child_module, 'dis_points_norm'):
            tmp_module_dis = child_module.dis_points_norm
            for i in range(len(tmp_module_dis)):
                dis_channels_norm.append(tmp_module_dis[i])
        if hasattr(child_module, 'hrank_channel_norm'):
            tmp_module_hrank_norm = child_module.hrank_channel_norm
            for i in range(len(tmp_module_hrank_norm)):
                hrank_channels_norm.append(tmp_module_hrank_norm[i])
    tmp_cd_hrank_normal = []
    for i in range(len(cof_channels_norm)):
        logging.info('cof_channel_norm is {}\n'.format(cof_channels_norm[i]))
        logging.info('dis_points_norm is {}\n'.format(dis_channels_norm[i]))
        logging.info('hrank_channels_norm is {}\n'.format(hrank_channels_norm[i]))
        tmp_cd_hrank_normal.append(
            cof_factor * cof_channels_norm[i].cpu().numpy() + dis_factor * dis_channels_norm[
                i].cpu().numpy() + rank_factor * hrank_channels_norm[i].numpy())
    if 'PointNet2Encoder' in cfg.model.encoder_args.NAME:
        cd_hrank_normal = tmp_cd_hrank_normal
        factor_list = [str(cof_factor), str(dis_factor), str(rank_factor)]
        base_name = 'pointnet++/' + '_'.join(factor_list) + '/' + str(cfg.model.dataset)
    else:
        cd_hrank_normal = [tmp_cd_hrank_normal[0], tmp_cd_hrank_normal[2], tmp_cd_hrank_normal[1],
                           tmp_cd_hrank_normal[2],
                           tmp_cd_hrank_normal[4],
                           tmp_cd_hrank_normal[3], tmp_cd_hrank_normal[4], tmp_cd_hrank_normal[6],
                           tmp_cd_hrank_normal[5],
                           tmp_cd_hrank_normal[6],
                           tmp_cd_hrank_normal[8], tmp_cd_hrank_normal[7], tmp_cd_hrank_normal[8],
                           tmp_cd_hrank_normal[9],
                           tmp_cd_hrank_normal[10]]
        factor_list = [str(cof_factor), str(dis_factor), str(rank_factor)]
        base_name = 'pointnext_' + str(cfg.model.encoder_args.width) + '/' + '_'.join(factor_list) + '/'
    rank_dir = '/data/private/project/pc/cls/PointNeXt/cd_rank/' + base_name + str(cfg.model.dataset)

    if not os.path.isdir(rank_dir):
        os.makedirs(rank_dir)

    for i in range(len(cd_hrank_normal)):
        np.save(rank_dir
                + '/conv_feature_rank_' + str(i) + '.npy',
                cd_hrank_normal[i])

    print('h_rank have save to {}!'.format(rank_dir))


def test(model, val_loader, validate_fn, epoch, cfg):
    macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
    print_cls_results(oa, macc, accs, epoch, cfg)


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        points = data['x']
        points = points[:, :npoints]
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    tp, count = cm.tp, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    return macc, overallacc, accs, cm
