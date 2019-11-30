from tensorboardX import SummaryWriter
tb_logger = SummaryWriter(log_dir='../tb_logger/1')
tb_logger.add_scalar('loss', 0.5, 1)