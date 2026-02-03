import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from geoopt.optim import RiemannianAdam
from mne.decoding import CSP
from sklearn.model_selection import KFold
import dataloader
import utils
from model.model import HE_MLPO
from utils import setup_seed, build_loader, save_model



# ==========================
# Environment / Global Config
# ==========================
os.environ["TRITON_F32_DEFAULT"] = "ieee"
os.environ["TRITON_PRINT_INFO"] = "0"
os.environ["TRITON_SILENT"] = "1"


# ==========================
# Loss Functions
# ==========================
class CenterLoss(nn.Module):
    """
    Center loss implemented using class prototypes as centers.
    """

    def __init__(self, num_classes: int, feat_dim: int, lambda_param: float = 1.0):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, features, labels, proto_0, proto_1):
        centers = torch.cat((proto_0, proto_1), dim=0)  # [2, D]
        batch_size = features.size(0)
        centers_batch = centers[labels]  # [B, D]
        loss = 0.5 * torch.sum((features - centers_batch) ** 2) / batch_size
        return self.lambda_param * loss


def hyperbolic_loss(x_hyp, proto0, proto1, y, manifold, center_loss_fn):
    """
    Hyperbolic classification loss:
    - CE loss using negative hyperbolic distance as logits
    - triplet-style contrastive margin
    - prototype separation constraint
    """
    d0 = manifold.dist(x_hyp, proto0)
    d1 = manifold.dist(x_hyp, proto1)

    pos_dist = torch.where(y == 0, d0, d1)
    neg_dist = torch.where(y == 0, d1, d0)
    cont_loss = F.relu(pos_dist - neg_dist + 1.0).mean()

    proto_dist = manifold.dist(proto0, proto1)
    sep_loss = F.relu(2.0 - proto_dist).mean()

    c_loss = center_loss_fn(x_hyp, y, proto0, proto1)
    return 0.5 * cont_loss + 0.3 * sep_loss + 0.2*c_loss


# ==========================
# Train / Test
# ==========================
def train_one_epoch(model, source_loader, target_loader, optimizer, optimizer_h, center_loss_fn, device):
    model.train()

    correct = 0
    total = 0
    total_loss_meter = 0.0

    # NOTE: original code zips source and target
    for (data_s, label_s), (data_t, _) in zip(source_loader, target_loader):
        data_s = data_s.to(device).type(torch.cuda.FloatTensor)
        label_s = label_s.to(device)

        # Forward
        feature, proto1, proto2, manifold, adj = model(data_s)

        # Loss
        loss_hyp = hyperbolic_loss(feature, proto1, proto2, label_s.squeeze(1).long(), manifold,center_loss_fn)
        total_loss = loss_hyp

        # logits from hyperbolic distance
        proto = torch.cat((proto1, proto2), dim=0)
        distances = manifold.dist(feature.unsqueeze(1), proto)
        logits = -distances

        # Stats
        total_loss_meter += total_loss.item()
        total += data_s.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == label_s.squeeze(1)).sum().item()

        # Backward
        optimizer.zero_grad()
        optimizer_h.zero_grad()
        total_loss.backward()
        optimizer.step()
        optimizer_h.step()

    train_acc = 100.0 * correct / total
    avg_loss = total_loss_meter / max(1, len(source_loader))
    print(f"[Train] loss={avg_loss:.4f} | acc={train_acc:.2f}%")




@torch.no_grad()
def evaluate(model, loader, device, epoch=0,evaluate = False):
    model.eval()

    correct = 0
    total = 0

    label_list = []
    pred_list = []
    feature_list = []
    adj_list = []

    for data, label in loader:
        data = data.to(device).type(torch.cuda.FloatTensor)
        label = label.to(device)

        feature, proto1, proto2, manifold, adj = model(data)

        proto = torch.cat((proto1, proto2), dim=0)
        distances = manifold.dist(feature.unsqueeze(1), proto)
        logits = -distances
        preds = torch.argmax(logits, dim=1)

        correct += (preds == label.squeeze(1)).sum().item()
        total += logits.size(0)

        label_list.append(label)
        pred_list.append(preds)
        feature_list.append(feature)
        adj_list.append(adj)

    acc = 100.0 * correct / total

    label_list = torch.cat(label_list, dim=0)
    pred_list = torch.cat(pred_list, dim=0)
    feature_list = torch.cat(feature_list, dim=0)
    adj_list = torch.cat(adj_list, dim=0)

    recall, f1, precision = utils.compute_direct(label_list.cpu(), pred_list.cpu())
    if evaluate:
        print(
            f"[Test] epoch={epoch} | acc={acc:.2f}% | recall={recall:.2f}% | f1={f1:.2f}% | precision={precision:.2f}%"
        )

    return acc, recall, f1, precision, feature_list, label_list, adj_list


# ==========================
# Data Loading
# ==========================
def get_data(subject_name="S1", time_len=1, dataset="KUL"):
    """
    Wrapper for your existing dataloader.
    """
    DTU_document_path = r"D:\数据集\DTU\DATA_preproc128-2"
    if dataset == "DTU":
        return dataloader.get_DTU_trail_data(subject_name, time_len, DTU_document_path)

# ==========================
# Main
# ==========================
def main():
    # --------------------------
    # Config
    # --------------------------
    time_len = 1
    dataset = "DTU"

    epochs = 100
    lr = 3e-4
    seed = 2026
    patience = 20

    subject_ranges = [1,2,3,4,5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(seed)

    # logging
    os.makedirs(f"./accuracy/", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - Subject: %(subject_id)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"./accuracy/{time_len}s.log",
        filemode="a",
    )

    # input dimension
    in_dim = math.ceil(128 * time_len)
    save_dir = f"./saved_models/{time_len}s"
    print("准备创建目录:", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("目录是否存在:", os.path.exists(save_dir))
    print("当前工作目录:", os.getcwd())
    # --------------------------
    # Run per subject
    # --------------------------
    for subj in subject_ranges:
        subject_name = f"S{subj}"
        print("\n" + "=" * 60)
        print(f"[Subject] {subject_name} | dataset={dataset} | time={time_len}s")
        print("=" * 60)

        eeg_data, event_data = get_data(subject_name, time_len, dataset)

        kf = KFold(n_splits=4, shuffle=True, random_state=42)

        for fold, (train_ids, test_ids) in enumerate(kf.split(eeg_data)):
            print("\n" + "-" * 50)
            print(f"FOLD {fold}")
            print("-" * 50)
            print("train_index:", train_ids)
            print("test_index:", test_ids)
            # split
            train_data = eeg_data[train_ids]
            train_event = event_data[train_ids]
            test_data = eeg_data[test_ids]
            test_event = event_data[test_ids]

            # CSP
            csp = CSP(
                n_components=64,
                reg=None,
                log=None,
                cov_est="concat",
                transform_into="csp_space",
                norm_trace=True,
            )
            train_eeg = csp.fit_transform(train_data.transpose(0, 2, 1), train_event).transpose(0, 2, 1)
            test_eeg = csp.transform(test_data.transpose(0, 2, 1)).transpose(0, 2, 1)

            train_loader = build_loader(train_eeg, train_event, time_len, 64, batch_size=32)
            test_loader = build_loader(test_eeg, test_event, time_len, 64, batch_size=32)

            # model
            model = HE_MLPO(64, in_dim, sample_rate=128, layer=4).to(device)
            center_loss_fn = CenterLoss(num_classes=2, feat_dim=128)

            # param groups (keep original logic)
            euclidean_params = []
            hyperbolic_params = []
            for name, param in model.named_parameters():
                if "hyp_" in name or "mapper" in name:
                    hyperbolic_params.append(param)
                else:
                    euclidean_params.append(param)

            optimizer = optim.Adam(
                [{"params": model.parameters()}, {"params": center_loss_fn.parameters()}],
                weight_decay=0.001,
                lr=lr,
            )

            optimizer_h = RiemannianAdam(
                [
                    {"params": euclidean_params, "lr": 1e-4, "stabilize": None},
                    {"params": hyperbolic_params, "lr": 1e-3},
                ]
            )

            # training loop
            best_acc = 0.0
            best_metrics = (0.0, 0.0, 0.0)  # recall, f1, precision
            early_stop_count = 0

            for epoch in range(1, epochs + 1):
                train_one_epoch(
                    model=model,
                    source_loader=train_loader,
                    target_loader=test_loader,
                    optimizer=optimizer,
                    optimizer_h=optimizer_h,
                    center_loss_fn=center_loss_fn,
                    device=device,
                )

                acc, recall, f1, precision, feature_list, label_list, adj_list = evaluate(
                    model=model,
                    loader=test_loader,
                    device=device,
                    epoch=epoch,
                )

                if acc > best_acc:
                    best_acc = acc
                    best_metrics = (recall, f1, precision)

                    save_path = os.path.join(save_dir, f"s{subj}.pth")
                    save_model(model, optimizer, epoch + 1, save_path)

                    print(f"[Save] Best model updated at epoch={epoch} -> {save_path}")
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= patience:
                        print(f"[Early Stop] No improvement for {patience} epochs.")
                        break

            # logging
            recall, f1, precision = best_metrics
            log_message = (
                f"subject{subj} : Accuracy={best_acc:.2f}% || Recall={recall:.2f}% || "
                f"F1={f1:.2f}% || Precision:{precision:.2f}%"
            )
            logging.info(log_message, extra={"subject_id": subj})

            # reload best and visualize chord
            best_model_path = os.path.join(save_dir, f"s{subj}.pth")
            test_model, _ = utils.load_model(path=best_model_path, model=model)
            acc, recall, f1, precision, feature_list, label_list, adj_list = evaluate(
                model=test_model,
                loader=test_loader,
                device=device,
                epoch=1,
                evaluate=True
            )

        del eeg_data, event_data, train_loader, test_loader


if __name__ == "__main__":
    main()
