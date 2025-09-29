import os
import time
import warnings
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import yaml
from exp.exp_basic import Exp_Basic
from models import MODEL_REQUIRES_CYCLE
from torch.func import functional_call
from torch.utils.data import DataLoader
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.tools import EarlyStopping, Scheduler, clip_grads, disable_grad, enable_grad, log_heatmap, plot_heatmap, \
    split_dataset, split_dataset_with_overlap, visual

warnings.filterwarnings('ignore')


class CovarianceMatrix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.eps = 1e-6
        self.auxi_loss = args.auxi_loss
        self.meta_type = getattr(args, 'meta_type', 'all')

        if self.meta_type == 'all':
            self.L_param = nn.Parameter(torch.eye(args.pred_len))
        elif self.meta_type == 'diag':
            self.diag_param = nn.Parameter(torch.ones(args.pred_len))
        elif self.meta_type == 'off_diag':
            self.L_param = nn.Parameter(torch.zeros(args.pred_len, args.pred_len))
        else:
            raise ValueError(f"Unknown meta_type: {self.meta_type}. Supported types: ['all', 'diag', 'off_diag']")

    def _get_L(self, params=None):
        if not hasattr(self, 'meta_type'):
            self.meta_type = 'all'

        if self.meta_type == 'all':
            # 原始模式：完整的Cholesky分解
            if params is None:
                L_param = self.L_param
            else:
                L_param = params['L_param']
            
            # 取下三角并在对角线加 eps，确保正定
            L = torch.tril(L_param)
            diag = torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1) + self.eps)
            L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + diag
            return L
            
        elif self.meta_type == 'diag':
            # 对角线模式：L矩阵是对角矩阵
            if params is None:
                diag_param = self.diag_param
            else:
                diag_param = params['diag_param']
            
            # 对角矩阵的L就是对角线元素的平方根，确保正值
            diag_values = torch.sqrt(torch.abs(diag_param) + self.eps)
            return torch.diag(diag_values)
            
        elif self.meta_type == 'off_diag':
            # 非对角线模式：对角线固定为1，只学习下三角非对角线部分
            if params is None:
                L_param = self.L_param
            else:
                L_param = params['L_param']
            
            # 只取严格下三角部分（不包括对角线），对角线固定为1
            L = torch.tril(L_param, diagonal=-1)
            L = L + torch.eye(self.pred_len, device=L.device, dtype=L.dtype)

            # 对每一行进行归一化，确保A=L@L.T的对角线为1
            row_norms = torch.norm(L, dim=1, keepdim=True)
            row_norms = torch.clamp(row_norms, min=self.eps)  # 避免除零
            L = L / row_norms
            return L

    def forward(self, params=None):
        L = self._get_L(params)
        return L @ L.transpose(-1, -2)               # Σ = L Lᵀ

    def get_inverse(self, params=None):
        L = self._get_L(params)
        A = L @ L.transpose(-1, -2)
        return torch.linalg.inv(A)

    def get_loss(self, pred, target, params=None):
        """ML3 Learned Loss Function"""
        L = self._get_L(params)  # [P, P] 下三角矩阵

        E = pred - target  # [B, P, D]
        E_flat = E.permute(0, 2, 1).reshape(-1, self.pred_len)  # [B*D, P]

        if self.meta_type == 'diag':
            # 对于对角矩阵，可以直接进行元素级除法
            diag_inv = 1.0 / torch.diag(L)  # L的对角线元素的倒数
            x = E_flat * diag_inv.unsqueeze(0)  # 广播乘法
        else:
            # 解线性方程组 Lx = E_flat，得到 x = L^{-1}E_flat
            # 使用三角求解器（L是下三角矩阵）
            x = torch.linalg.solve_triangular(
                L, 
                E_flat.T,  # 转置为 [P, B*D]
                upper=False, 
                unitriangular=False
            ).T  # 转置回 [B*D, P]

        # 计算二次型: x^T x
        if self.auxi_loss == 'MSE':
            loss = torch.mean(x ** 2)
        elif self.auxi_loss == 'MAE':
            loss = torch.mean(x.abs())
        else:
            raise AttributeError(f"No defined loss type for {self.auxi_loss}.")

        if self.args.reg_lambda > 0 and self.meta_type in ['all', 'off_diag']:
            Sigma = self.forward(params)  # [P, P]
            off_diag = Sigma - torch.diag_embed(torch.diagonal(Sigma))
            reg_loss = torch.norm(off_diag, p='fro') ** 2
            loss += self.args.reg_lambda * reg_loss

        return loss


def get_projection(A):
    with torch.no_grad():
        Am = A()
    return Am.detach().cpu().numpy()  # 返回 numpy 数组


def get_param_dict(module):
    # 返回 OrderedDict，适用于 functional forward
    return dict(module.named_parameters())


def update_param_dict(param_dict, grads_dict, lr):
    """
    使用梯度字典更新参数字典。
    grads_dict 仅包含有梯度的参数，其他参数视为 grad=0（保持不变）。
    """
    updated_params = {}
    for k, v in param_dict.items():
        if k in grads_dict and grads_dict[k] is not None:
            updated_params[k] = v - lr * grads_dict[k]
        else:
            # 该参数无梯度 → 视为梯度为 0 → 不更新
            updated_params[k] = v  # 保持原参数，等价于 +0
    return updated_params


class Exp_Long_Term_Forecast_META_ML3(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.n_inner = args.meta_inner_steps
        self.lr = args.learning_rate
        self.inner_lr = args.inner_lr
        self.meta_lr = args.meta_lr
        self.first_order = args.first_order
        self.model_per_task = args.model_per_task
        self.num_tasks = args.num_tasks

        self.A = CovarianceMatrix(self.args).to(self.device)
        self.task_models = [self.model]
        if self.model_per_task:
            for _ in range(1, self.num_tasks):
                task_model = self._build_model().to(self.device)
                self.task_models.append(task_model)
        else:
            self.task_models = [self.model] * self.num_tasks

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_cov_loss = []
        self.model.eval()
        self.A.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)  # 标准损失
                cov_loss = self.A.get_loss(pred, true)  # 学习到的损失

                total_loss.append(loss)
                total_cov_loss.append(cov_loss)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        total_loss = torch.mean(torch.stack(total_loss)).item()
        total_cov_loss = torch.mean(torch.stack(total_cov_loss)).item()

        self.model.train()
        self.A.train()
        return total_loss, total_cov_loss

    def inner_loop(self, task_id, support_loader, query_loader):
        # 获取当前模型参数（每个meta epoch都从当前状态开始，而不是初始状态）
        task_model = self.task_models[task_id]
        model_params_init = get_param_dict(task_model)

        # 内层循环：使用学习到的损失函数训练模型参数
        fast_model_params = {k: v.clone() for k, v in model_params_init.items()}
        for k in range(self.n_inner):
            bx, by, bx_mark, by_mark, by_cycle = next(support_loader)
            outputs, batch_y, _ = self.forward_step_with_params(
                bx, by, bx_mark, by_mark, by_cycle, fast_model_params, task_model
            )
            loss = self.A.get_loss(outputs, batch_y)

            model_grads = torch.autograd.grad(
                loss, fast_model_params.values(), 
                create_graph=not self.first_order, 
                allow_unused=True
            )
            model_grads = clip_grads(model_grads, self.args.max_norm)
            model_grads_dict = {k: g for k, g in zip(fast_model_params.keys(), model_grads)}
            fast_model_params = update_param_dict(fast_model_params, model_grads_dict, self.inner_lr)

        # 外层循环：在query set上使用标准损失评估性能
        bx, by, bx_mark, by_mark, by_cycle = next(query_loader)
        outputs, batch_y, _ = self.forward_step_with_params(
            bx, by, bx_mark, by_mark, by_cycle, fast_model_params, task_model
        )
        meta_loss = self.A.get_loss(outputs, batch_y)
        return meta_loss

    def forward_step_with_params(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, params, model):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE']:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        model_args = [batch_x, batch_x_mark, dec_inp, batch_y_mark]
        if self.args.model in MODEL_REQUIRES_CYCLE:
            model_args.append(batch_cycle)
        model_args = tuple(model_args)
        if self.args.output_attention:
            outputs, attn = functional_call(model, params, model_args)
        else:
            outputs, attn = functional_call(model, params, model_args), None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.pred_len:, f_dim:]
        return outputs, batch_y, attn

    def initialize_meta_tasks(self, train_data):
        self.meta_learning = False

        task_data_list = split_dataset_with_overlap(train_data, self.num_tasks, self.args.overlap_ratio)
        task_data_list = [split_dataset(task_data, r=0.7) for task_data in task_data_list]

        support_data_list = [td[0] for td in task_data_list]
        support_loader_list = [DataLoader(support_data, batch_size=self.args.auxi_batch_size, shuffle=True) for support_data in support_data_list]
        support_loader_list = [cycle(support_loader) for support_loader in support_loader_list]

        query_data_list = [td[1] for td in task_data_list]
        query_loader_list = [DataLoader(query_data, batch_size=self.args.auxi_batch_size, shuffle=True) for query_data in query_data_list]
        query_loader_list = [cycle(query_loader) for query_loader in query_loader_list]
        return support_loader_list, query_loader_list

    def meta_train(self, support_loader_list, query_loader_list, path, res_path):
        # 在meta train阶段，损失函数参数可训练，模型参数也需要梯度（用于inner loop）
        enable_grad(self.A)
        enable_grad(self.model)

        A_optim = self._select_optimizer(self.A, self.meta_lr, optim_type=self.args.meta_optim_type)
        A_scheduler = Scheduler(A_optim, self.args, self.args.warmup_steps)

        epoch_time = time.time()
        meta_step = 0
        for step in range(self.args.warmup_steps):
            meta_step = step + 1
            verbose = (meta_step % 100 == 0)
            task_losses = []

            meta_lr_cur = A_scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/meta_train/meta_lr', meta_lr_cur, meta_step)

            self.model.train()
            self.A.train()

            # 遍历所有任务，累积meta loss
            for task_id, (support_loader, query_loader) in enumerate(zip(support_loader_list, query_loader_list)):
                meta_loss = self.inner_loop(task_id, support_loader, query_loader)
                task_losses.append(meta_loss)
                self.writer.add_scalar(f'{self.pred_len}/meta_train/task_{task_id+1}_meta_loss', meta_loss.item(), meta_step)
                if verbose:
                    print(f"\ttask: {task_id + 1}, total task: {self.num_tasks} | meta loss: {meta_loss.item():.7f}")

            # 统一进行损失函数参数的更新
            A_optim.zero_grad()
            avg_meta_loss = torch.stack(task_losses).mean()
            avg_meta_loss.backward()
            A_optim.step()

            avg_meta_loss_val = avg_meta_loss.item()
            self.writer.add_scalar(f'{self.pred_len}/meta_train/meta_loss', avg_meta_loss_val, meta_step)
            log_heatmap(self.writer, get_projection(self.A), f'{self.pred_len}/cov_mat', meta_step)

            if verbose:
                print(f"Step: {meta_step} cost time: {time.time() - epoch_time:.2f}s")
                print(f"Step: {meta_step} | Avg Meta Loss: {avg_meta_loss_val:.7f}")
                epoch_time = time.time()

            if self.args.lradj in ['TST']:
                A_scheduler.step(verbose=verbose)
            else:
                if verbose:
                    A_scheduler.step(avg_meta_loss_val, meta_step // 100)

        best_A_path = os.path.join(path, 'A.pth')
        torch.save(self.A, best_A_path)
        projection = get_projection(self.A)
        plot_heatmap(projection, save_path=os.path.join(res_path, 'cov_matrix.pdf'))
        print(f'Statistics of learned covariance matrix: \033[92m(mean: {np.mean(projection):.4f}, std: {np.std(projection):.4f}, min: {np.min(projection):.4f}, max: {np.max(projection):.4f})\033[0m')

    def meta_test(self, train_loader, vali_data, vali_loader, criterion, path):
        if self.model_per_task and self.num_tasks > 1:
            del self.task_models[1:]

        disable_grad(self.A)
        enable_grad(self.model)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer(self.model, self.lr)
        scheduler = Scheduler(model_optim, self.args, train_steps)

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss, train_loss_mse = [], []

            lr_cur = scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/meta_test/lr', lr_cur, self.epoch)

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.model.train()
                self.A.eval()

                self.step += 1
                iter_count += 1

                model_optim.zero_grad()
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                loss = self.A.get_loss(outputs, batch_y)
                loss.backward()
                model_optim.step()

                with torch.no_grad():
                    loss_mse = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                train_loss_mse.append(loss_mse.item())
                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss', loss.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss_mse', loss_mse.item(), self.step)

                if (i + 1) % 100 == 0:
                    print(f"\tMeta Test - iters: {i + 1}, epoch: {self.epoch} | loss: {loss.item():.7f}, mse loss: {loss_mse.item():.7f}")
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(train_loader) - i)
                    print(f'\tspeed: {speed:.4f}s/iter; cost time: {cost_time:.4f}s; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_mse = np.average(train_loss_mse)
            valid_loss_mse, valid_loss_cov = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/meta_test/loss_cov', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta_test/loss_mse', train_loss_mse, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss_cov', valid_loss_cov, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss_mse', valid_loss_mse, self.epoch)

            print(f"Epoch: {self.epoch} | Train Loss Cov: {train_loss:.7f}, MSE: {train_loss_mse:.7f} | Valid Loss Cov: {valid_loss_cov:.7f}, MSE: {valid_loss_mse:.7f}")
            early_stopping(valid_loss_mse, self.model, path)
            if early_stopping.early_stop:
                print("Meta Test Early stopping")
                break

            if self.args.lradj not in ['TST']:
                scheduler.step(valid_loss_mse, self.epoch)

    def train(self, setting, prof=None):
        train_data, train_loader = self._get_data(flag='train')
        support_loader_list, query_loader_list = self.initialize_meta_tasks(train_data)
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        self.writer = self._create_writer(res_path)

        criterion = self._select_criterion()

        # ============ Meta Train 阶段：只训练损失函数 ============
        print("\n>>>>>>>Meta Training Phase\n")
        self.meta_train(support_loader_list, query_loader_list, path, res_path)
        print("\n>>>>>>>Meta Training Phase completed\n")

        # ============ ML3 Meta Test 阶段：重新初始化模型，使用学习到的损失函数训练 ============
        print("\n>>>>>>>Meta Test Phase\n")
        self.meta_test(train_loader, vali_data, vali_loader, criterion, path)
        print("\n>>>>>>>Meta Test Phase completed\n")

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        best_A_path = os.path.join(path, 'A.pth')
        self.A = torch.load(best_A_path)

        return self.model

    def test(self, setting, prof=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
            self.A = torch.load(os.path.join(ckpt_dir, 'A.pth'))

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        self.A.eval()
        # metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                batch_x = batch_x.detach()
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_data.scale and self.args.inverse:
                    batch_x = batch_x.cpu().numpy()
                    in_shape = batch_x.shape
                    batch_x = test_data.inverse_transform(batch_x.reshape(-1, in_shape[-1])).reshape(in_shape)
                    batch_x = torch.from_numpy(batch_x).float().to(self.device)

                    outputs = outputs.cpu().numpy()
                    batch_y = batch_y.cpu().numpy()
                    out_shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, out_shape[-1])).reshape(out_shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, out_shape[-1])).reshape(out_shape)
                    outputs = torch.from_numpy(outputs).float().to(self.device)
                    batch_y = torch.from_numpy(batch_y).float().to(self.device)

                inputs.append(batch_x.cpu())
                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())

                if i % 20 == 0 and self.output_vis:
                    gt = np.concatenate((batch_x[0, :, -1].cpu().numpy(), batch_y[0, :, -1].cpu().numpy()), axis=0)
                    pd = np.concatenate((batch_x[0, :, -1].cpu().numpy(), outputs[0, :, -1].cpu().numpy()), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        inputs = torch.cat(inputs, dim=0)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        print('test shape:', preds.shape, trues.shape)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        # m = metric_collector.compute()
        # mae, mse, rmse, mape, mspe, mre = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"], m["mre"]
        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        with torch.no_grad():
            self.A.to(preds.device)
            cov_loss = self.A.get_loss(preds, trues)
        print('{}\t| mse:{}, mae:{}, cov:{}'.format(self.pred_len, mse, mae, cov_loss))

        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mre', mre, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/cov', cov_loss, self.epoch)
        self.writer.close()

        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}, cov:{}'.format(mse, mae, cov_loss))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, cov_loss, rmse, mape, mspe, mre]))

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())

        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            args_dict = vars(self.args)
            with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

        return
