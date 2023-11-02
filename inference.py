def inference(
    args,
    dlrm,
    best_acc_test,
    best_auc_test,
    test_ld,
    device,
    use_gpu,
    log_iter=-1,
):
    test_accu = 0
    test_samp = 0

    if args.mlperf_logging:
        scores = []
        targets = []

    for i, testBatch in enumerate(test_ld):
        # early exit if nbatches was set by the user and was exceeded
        if nbatches > 0 and i >= nbatches:
            break

        X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
            testBatch
        )

        # Skip the batch if batch size not multiple of total ranks
        if ext_dist.my_size > 1 and X_test.size(0) % ext_dist.my_size != 0:
            print("Warning: Skiping the batch %d with size %d" % (i, X_test.size(0)))
            continue

        # forward pass
        Z_test = dlrm_wrap(
            X_test,
            lS_o_test,
            lS_i_test,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        ### gather the distributed results on each rank ###
        # For some reason it requires explicit sync before all_gather call if
        # tensor is on GPU memory
        if Z_test.is_cuda:
            torch.cuda.synchronize()
        (_, batch_split_lengths) = ext_dist.get_split_lengths(X_test.size(0))
        if ext_dist.my_size > 1:
            Z_test = ext_dist.all_gather(Z_test, batch_split_lengths)

        if args.mlperf_logging:
            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array
            scores.append(S_test)
            targets.append(T_test)
        else:
            with record_function("DLRM accuracy compute"):
                # compute loss and accuracy
                S_test = Z_test.detach().cpu().numpy()  # numpy array
                T_test = T_test.detach().cpu().numpy()  # numpy array

                mbs_test = T_test.shape[0]  # = mini_batch_size except last
                A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

                test_accu += A_test
                test_samp += mbs_test

    if args.mlperf_logging:
        with record_function("DLRM mlperf sklearn metrics compute"):
            scores = np.concatenate(scores, axis=0)
            targets = np.concatenate(targets, axis=0)

            metrics = {
                "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "ap": sklearn.metrics.average_precision_score,
                "roc_auc": sklearn.metrics.roc_auc_score,
                "accuracy": lambda y_true, y_score: sklearn.metrics.accuracy_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
            }

        validation_results = {}
        for metric_name, metric_function in metrics.items():
            validation_results[metric_name] = metric_function(targets, scores)
            writer.add_scalar(
                "mlperf-metrics-test/" + metric_name,
                validation_results[metric_name],
                log_iter,
            )
        acc_test = validation_results["accuracy"]
    else:
        acc_test = test_accu / test_samp
        writer.add_scalar("Test/Acc", acc_test, log_iter)

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
        "state_dict": dlrm.state_dict(),
        "test_acc": acc_test,
    }

    if args.mlperf_logging:
        is_best = validation_results["roc_auc"] > best_auc_test
        if is_best:
            best_auc_test = validation_results["roc_auc"]
            model_metrics_dict["test_auc"] = best_auc_test
        print(
            "recall {:.4f}, precision {:.4f},".format(
                validation_results["recall"],
                validation_results["precision"],
            )
            + " f1 {:.4f}, ap {:.4f},".format(
                validation_results["f1"], validation_results["ap"]
            )
            + " auc {:.4f}, best auc {:.4f},".format(
                validation_results["roc_auc"], best_auc_test
            )
            + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                validation_results["accuracy"] * 100, best_acc_test * 100
            ),
            flush=True,
        )
    else:
        is_best = acc_test > best_acc_test
        if is_best:
            best_acc_test = acc_test
        print(
            " accuracy {:3.3f} %, best {:3.3f} %".format(
                acc_test * 100, best_acc_test * 100
            ),
            flush=True,
        )
    return model_metrics_dict, is_best
