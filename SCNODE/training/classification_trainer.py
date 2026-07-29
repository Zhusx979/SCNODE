import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.metrics import classification_report

from blood_experiment.cam import generate_cam_overlays
from blood_experiment.evaluation import build_evaluation_bundle, save_evaluation_artifacts
from blood_experiment.visualization import (
    save_confusion_matrix_plot,
    save_precision_recall_plot,
    save_roc_curve_plot,
    save_training_history_plots,
)
from SCNODE.training.experiment_config import args
from SCNODE.training.ode_runtime import get_and_reset_ode_nfe
from SCNODE.training.progress_reporting import (
    append_epoch_metrics_row,
    build_epoch_summary_lines,
    format_seconds,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        if hasattr(m, "weight"):
            init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias, 0)
    elif class_name.find("BatchNorm") != -1:
        if hasattr(m, "weight"):
            init.constant_(m.weight, 1)
        if hasattr(m, "bias"):
            init.constant_(m.bias, 0)


def lr_schedule(lr, epoch):
    decay_factor = 0
    if epoch >= 15:
        decay_factor = 3
    elif epoch >= 10:
        decay_factor = 2
    elif epoch >= 5:
        decay_factor = 1
    return lr / math.pow(10, decay_factor)


def _unpack_batch(batch):
    if len(batch) == 3:
        inputs, labels, paths = batch
        return inputs, labels, list(paths)
    if len(batch) == 2:
        inputs, labels = batch
        return inputs, labels, [""] * len(labels)
    raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")


def _current_learning_rate(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _batch_status(step: int, total_steps: int, avg_loss: float, accuracy: float, learning_rate: float) -> str:
    return (
        f"step {step}/{total_steps} | "
        f"loss={avg_loss:.4f} | acc={accuracy:.2f}% | lr={learning_rate:.6g}"
    )


def _progress_iter(loader, desc: str):
    if args.use_tqdm and tqdm is not None:
        return tqdm(loader, total=len(loader), desc=desc, leave=False, dynamic_ncols=True)
    return loader


def _progress_update(progress, text: str) -> None:
    if args.use_tqdm and tqdm is not None and hasattr(progress, "set_postfix_str"):
        progress.set_postfix_str(text)


def _progress_close(progress) -> None:
    if args.use_tqdm and tqdm is not None and hasattr(progress, "close"):
        progress.close()


def save_prediction_arrays(model_dir: Path, name: str, probabilities, labels, predictions) -> None:
    np.save(model_dir / f"{name}_predict.npy", np.asarray(probabilities))
    np.save(model_dir / f"{name}_label.npy", np.asarray(labels))
    np.save(model_dir / f"{name}_pred_label.npy", np.asarray(predictions))

def save_epoch_summary(
    history_path: Path,
    epoch: int,
    num_epochs: int,
    summary: dict,
    val_accuracy: float | None,
    test_accuracy: float,
    nfe_history: float = 0.0,
    bnfe_history: float = 0.0,
) -> None:
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nEpoch [{epoch + 1}/{num_epochs}]\n")
        if val_accuracy is not None:
            handle.write(f"Validation Accuracy: {val_accuracy:.4f}%\n")
        handle.write(f"Test Accuracy: {test_accuracy:.4f}%\n")
        handle.write(f"Macro F1: {summary['macro_f1']:.6f}\n")
        handle.write(f"Balanced Accuracy: {summary['balanced_accuracy']:.6f}\n")
        handle.write(f"MCC: {summary['mcc']:.6f}\n")
        if args.is_ode:
            handle.write(f"NFE-F: {nfe_history:.4f}, NFE-B: {bnfe_history:.4f}\n")
        handle.write("-" * 60 + "\n")


def _collect_cam_candidates(
    batch_inputs: torch.Tensor,
    batch_labels: torch.Tensor,
    batch_predictions: torch.Tensor,
    batch_paths: list[str],
    class_names: list[str],
    counters: dict[int, int],
) -> list[dict]:
    selected = []
    limit = max(args.cam_samples_per_class, 0)
    if limit == 0:
        return selected

    for index in range(batch_labels.size(0)):
        true_index = int(batch_labels[index].item())
        if counters[true_index] >= limit:
            continue
        counters[true_index] += 1
        selected.append(
            {
                "input_tensor": batch_inputs[index].detach().cpu(),
                "image_path": batch_paths[index],
                "true_index": true_index,
                "predicted_index": int(batch_predictions[index].item()),
                "true_label": class_names[true_index],
                "predicted_label": class_names[int(batch_predictions[index].item())],
            }
        )
    return selected


def _export_final_artifacts(
    model,
    model_dir: Path,
    name: str,
    class_names: list[str],
    probabilities,
    labels,
    predictions,
    train_accuracies,
    val_accuracies,
    test_accuracies,
    train_losses,
    val_losses,
    cam_samples,
):
    metrics_dir = model_dir / "metrics"
    plots_dir = model_dir / "plots"
    cam_dir = model_dir / "cam"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_evaluation_bundle(
        y_true=np.asarray(labels),
        y_pred=np.asarray(predictions),
        y_prob=np.asarray(probabilities),
        class_names=class_names,
    )
    save_evaluation_artifacts(metrics_dir, bundle, class_names, prefix=name)

    if args.generate_visualizations:
        save_confusion_matrix_plot(
            output_path=plots_dir / f"{name}_confusion_matrix.png",
            confusion=bundle["confusion_matrix"],
            class_names=class_names,
            normalize=False,
            title=f"{name} Confusion Matrix",
        )
        save_confusion_matrix_plot(
            output_path=plots_dir / f"{name}_confusion_matrix_normalized.png",
            confusion=bundle["confusion_matrix"],
            class_names=class_names,
            normalize=True,
            title=f"{name} Normalized Confusion Matrix",
        )
        save_roc_curve_plot(
            output_path=plots_dir / f"{name}_roc_curve.png",
            roc_payload=bundle["roc"],
            class_names=class_names,
            title=f"{name} ROC Curves",
        )
        save_precision_recall_plot(
            output_path=plots_dir / f"{name}_precision_recall_curve.png",
            pr_payload=bundle["precision_recall"],
            class_names=class_names,
            title=f"{name} Precision-Recall Curves",
        )
        save_training_history_plots(
            output_dir=plots_dir,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            test_accuracies=test_accuracies,
            name=name,
            train_losses=train_losses,
            val_losses=val_losses,
        )

    save_prediction_arrays(model_dir, name, probabilities, labels, predictions)

    if args.generate_cam and cam_samples:
        try:
            generate_cam_overlays(
                model=model,
                samples=cam_samples,
                class_names=class_names,
                output_dir=cam_dir,
                target_layer_name=args.cam_target_layer or None,
                noise_samples=args.cam_noise_samples,
                noise_sigma=args.cam_noise_sigma,
            )
        except Exception as exc:
            cam_dir.mkdir(parents=True, exist_ok=True)
            (cam_dir / "cam_generation_error.txt").write_text(str(exc), encoding="utf-8")

    return bundle


def train_val_test_model(
    model,
    trainloader,
    valloader,
    testloader,
    criterion,
    device,
    name,
    class_names,
    num_epochs=20,
):
    model_dir = Path(args.folder_name) / name
    model_dir.mkdir(parents=True, exist_ok=True)
    history_path = model_dir / f"{name}_history.txt"
    epoch_metrics_csv_path = model_dir / f"{name}_epoch_metrics.csv"

    best_accuracy = 0.0
    best_test_accuracy = 0.0
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    train_losses = []
    val_losses = []

    if args.is_ode:
        epoch_nfes = 0
        epoch_backward_nfes = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_train_loss = 0.0
        correct = 0
        total = 0

        progress = _progress_iter(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]")
        for step_index, batch in enumerate(progress, start=1):
            inputs, labels, _ = _unpack_batch(batch)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if args.is_ode:
                nfe_forward = get_and_reset_ode_nfe(model)
                epoch_nfes += nfe_forward

            loss = criterion(outputs, labels)
            loss.backward()

            if args.is_ode:
                nfe_backward = get_and_reset_ode_nfe(model)
                epoch_backward_nfes += nfe_backward

            optimizer.step()

            running_train_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_accuracy = 100 * correct / total
            average_loss = running_train_loss / step_index
            learning_rate = _current_learning_rate(optimizer)
            batch_text = _batch_status(
                step=step_index,
                total_steps=len(trainloader),
                avg_loss=average_loss,
                accuracy=running_accuracy,
                learning_rate=learning_rate,
            )
            _progress_update(progress, batch_text)
            if (not args.use_tqdm or tqdm is None) and (
                step_index == 1
                or step_index == len(trainloader)
                or step_index % max(args.train_log_interval, 1) == 0
            ):
                print(f"[train] {batch_text}")
        _progress_close(progress)

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        train_losses.append(running_train_loss / len(trainloader))
        epoch_duration = time.time() - epoch_start_time

        avg_nfe = 0.0
        avg_bnfe = 0.0
        if args.is_ode:
            avg_nfe = epoch_nfes / len(trainloader)
            avg_bnfe = epoch_backward_nfes / len(trainloader)
            epoch_nfes = 0
            epoch_backward_nfes = 0

        val_accuracy = None
        val_loss_value = None
        if valloader is not None and len(valloader) > 0:
            model.eval()
            running_val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                val_progress = _progress_iter(valloader, desc=f"Epoch {epoch + 1}/{num_epochs} [val]")
                for step_index, batch in enumerate(val_progress, start=1):
                    inputs, labels, _ = _unpack_batch(batch)
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    predicted = outputs.argmax(dim=1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    _progress_update(
                        val_progress,
                        _batch_status(
                            step=step_index,
                            total_steps=len(valloader),
                            avg_loss=running_val_loss / step_index,
                            accuracy=100 * val_correct / val_total,
                            learning_rate=_current_learning_rate(optimizer),
                        ),
                    )
                _progress_close(val_progress)

            val_accuracy = 100 * val_correct / val_total
            val_loss_value = running_val_loss / len(valloader)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss_value)

            if val_accuracy > best_accuracy and args.save_report_pth:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), model_dir / f"{name}_best_model.pth")
        else:
            val_losses.append(float("nan"))

        model.eval()
        all_labels = []
        all_predictions = []
        all_probabilities = []
        cam_candidates = []
        cam_counters = defaultdict(int)

        with torch.no_grad():
            test_progress = _progress_iter(testloader, desc=f"Epoch {epoch + 1}/{num_epochs} [test]")
            test_seen = 0
            test_correct = 0
            for step_index, batch in enumerate(test_progress, start=1):
                inputs, labels, paths = _unpack_batch(batch)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                predicted = probabilities.argmax(dim=1)

                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                test_seen += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                _progress_update(
                    test_progress,
                    f"step {step_index}/{len(testloader)} | acc={100 * test_correct / test_seen:.2f}%",
                )

                if args.generate_cam:
                    cam_candidates.extend(
                        _collect_cam_candidates(
                            batch_inputs=inputs.cpu(),
                            batch_labels=labels.cpu(),
                            batch_predictions=predicted.cpu(),
                            batch_paths=paths,
                            class_names=class_names,
                            counters=cam_counters,
                        )
                    )
            _progress_close(test_progress)

        test_accuracy = 100 * np.mean(np.asarray(all_predictions) == np.asarray(all_labels))
        test_accuracies.append(test_accuracy)
        best_test_accuracy = max(best_test_accuracy, test_accuracy)

        report = classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        if args.full_report:
            print(report)

        bundle = build_evaluation_bundle(
            y_true=np.asarray(all_labels),
            y_pred=np.asarray(all_predictions),
            y_prob=np.asarray(all_probabilities),
            class_names=class_names,
        )
        skipped_images = sum(
            getattr(loader.dataset, "skipped_image_count", 0)
            for loader in (trainloader, valloader, testloader)
            if loader is not None
        )
        remaining_epochs = num_epochs - (epoch + 1)
        eta_seconds = remaining_epochs * epoch_duration
        summary_lines = build_epoch_summary_lines(
            epoch=epoch + 1,
            num_epochs=num_epochs,
            train_loss=train_losses[-1],
            train_accuracy=train_accuracy,
            val_loss=val_loss_value,
            val_accuracy=val_accuracy,
            test_accuracy=test_accuracy,
            macro_f1=bundle["summary"]["macro_f1"],
            balanced_accuracy=bundle["summary"]["balanced_accuracy"],
            mcc=bundle["summary"]["mcc"],
            learning_rate=_current_learning_rate(optimizer),
            epoch_duration=epoch_duration,
            eta_seconds=eta_seconds,
            best_val_accuracy=best_accuracy if valloader is not None and len(valloader) > 0 else None,
            best_test_accuracy=best_test_accuracy,
            skipped_images=skipped_images,
        )
        print("\n" + "=" * 72)
        for line in summary_lines:
            print(line)
        if args.is_ode:
            print(f"ODE    nfe_f={avg_nfe:.2f}  nfe_b={avg_bnfe:.2f}")
        if args.full_report:
            print(report)
        print("=" * 72)

        append_epoch_metrics_row(
            csv_path=epoch_metrics_csv_path,
            row={
                "epoch": epoch + 1,
                "train_loss": train_losses[-1],
                "train_accuracy": train_accuracy,
                "val_loss": "" if val_loss_value is None else val_loss_value,
                "val_accuracy": "" if val_accuracy is None else val_accuracy,
                "test_accuracy": test_accuracy,
                "macro_f1": bundle["summary"]["macro_f1"],
                "balanced_accuracy": bundle["summary"]["balanced_accuracy"],
                "mcc": bundle["summary"]["mcc"],
                "learning_rate": _current_learning_rate(optimizer),
                "epoch_duration_seconds": epoch_duration,
                "eta_seconds": eta_seconds,
                "best_val_accuracy": "" if val_accuracy is None else best_accuracy,
                "best_test_accuracy": best_test_accuracy,
                "skipped_images": skipped_images,
            },
        )

        if args.save_report_pth:
            save_epoch_summary(
                history_path=history_path,
                epoch=epoch,
                num_epochs=num_epochs,
                summary=bundle["summary"],
                val_accuracy=val_accuracy,
                test_accuracy=test_accuracy,
                nfe_history=avg_nfe,
                bnfe_history=avg_bnfe,
            )

            if epoch == num_epochs - 1:
                _export_final_artifacts(
                    model=model,
                    model_dir=model_dir,
                    name=name,
                    class_names=class_names,
                    probabilities=all_probabilities,
                    labels=all_labels,
                    predictions=all_predictions,
                    train_accuracies=train_accuracies,
                    val_accuracies=val_accuracies,
                    test_accuracies=test_accuracies,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    cam_samples=cam_candidates,
                )
