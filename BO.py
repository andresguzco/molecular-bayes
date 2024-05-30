from main_QM9 import *

def main(args):
    args.rank = 0
    is_main_process = (args.rank == 0)

    _log = FileLogger(is_master=is_main_process, is_rank0=is_main_process, output_dir=args.output_dir)
    _log.info(args)
    wandb.login()
    run = wandb.init(
    project="equiformer_v2",
    config=args,
    )
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ''' Dataset '''
    train_dataset = QM9(args.data_path, 'train', feature_type=args.feature_type)
    val_dataset   = QM9(args.data_path, 'valid', feature_type=args.feature_type)
    test_dataset  = QM9(args.data_path, 'test', feature_type=args.feature_type)
    _log.info('Training set mean: {}, std:{}'.format(
        train_dataset.mean(args.target), train_dataset.std(args.target)))
    
    # calculate dataset stats
    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean, task_std = train_dataset.mean(args.target), train_dataset.std(args.target)
    norm_factor = [task_mean, task_std]
    
    # since dataset needs random 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ''' Network '''
    create_model = registry.get_model_class(args.model_name)
    model = create_model(0, 0, 0)
    _log.info(model)
    model = model.to(device)
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = None #torch.nn.MSELoss() #torch.nn.L1Loss() # torch.nn.MSELoss() 
    if args.variational is True:
        criterion = det_loss
    elif args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError

    ''' AMP (from timm) '''
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
    
    ''' Data Loader '''
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem, 
            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    best_epoch, best_train_err, best_val_err, best_test_err = 0, float('inf'), float('inf'), float('inf')
    best_ema_epoch, best_ema_val_err, best_ema_test_err = 0, float('inf'), float('inf')
    
    for epoch in range(args.epochs):
        
        epoch_start_time = time.perf_counter()
        
        lr_scheduler.step(epoch)
        
        train_err = train_one_epoch(model=model, criterion=criterion, norm_factor=norm_factor,
            target=args.target, data_loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, model_ema=model_ema, 
            amp_autocast=amp_autocast, loss_scaler=loss_scaler,
            print_freq=args.print_freq, logger=_log, wandb=run, variational=args.variational)
        
        val_err, val_loss = evaluate(model, norm_factor, args.target, val_loader, device, 
            amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log,
            variational=args.variational, critFn=criterion)
        
        test_err, test_loss = evaluate(model, norm_factor, args.target, test_loader, device, 
            amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log, 
            variational=args.variational, critFn=criterion)
        wandb.log({
            "Validation Error": val_err,
            "Test Error": test_err,
        })
        
        # record the best results
        if val_err < best_val_err:
            best_val_err = val_err
            best_test_err = test_err
            best_train_err = train_err
            best_epoch = epoch

        info_str = 'Epoch: [{epoch}] Target: [{target}] train MAE: {train_mae:.5f}, '.format(
            epoch=epoch, target=args.target, train_mae=train_err)
        info_str += 'val MAE: {:.5f}, '.format(val_err)
        info_str += 'test MAE: {:.5f}, '.format(test_err)
        info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
        _log.info(info_str)
        
        info_str = 'Best -- epoch={}, train MAE: {:.5f}, val MAE: {:.5f}, test MAE: {:.5f}\n'.format(
            best_epoch, best_train_err, best_val_err, best_test_err)
        _log.info(info_str)
        
        # evaluation with EMA
        if model_ema is not None:
            ema_val_err, _ = evaluate(model_ema.module, norm_factor, args.target, val_loader, device, 
                amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log, 
                variational=args.variational, critFn=criterion)
            
            ema_test_err, _ = evaluate(model_ema.module, norm_factor, args.target, test_loader, device, 
                amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log, 
                variational=args.variational, critFn=criterion)
            
            # record the best results
            if (ema_val_err) < best_ema_val_err:
                best_ema_val_err = ema_val_err
                best_ema_test_err = ema_test_err
                best_ema_epoch = epoch
    
            info_str = 'Epoch: [{epoch}] Target: [{target}] '.format(
                epoch=epoch, target=args.target)
            info_str += 'EMA val MAE: {:.5f}, '.format(ema_val_err)
            info_str += 'EMA test MAE: {:.5f}, '.format(ema_test_err)
            info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
            _log.info(info_str)
            
            info_str = 'Best EMA -- epoch={}, val MAE: {:.5f}, test MAE: {:.5f}\n'.format(
                best_ema_epoch, best_ema_val_err, best_ema_test_err)
            _log.info(info_str)

        if epoch % 10 == 0:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': train_err,
                    }, os.path.join(os.getcwd(), filename(bayes='laplace' if args.laplace else 'VI' if args.variational else None)))
    
    if args.laplace is True:
        # We isolate the last layer for the Laplace Approximation
        last_layer_params = sum(p.numel() for p in model.energy_block.so3_linear_2.parameters())
        print(f'SO3 Linear Layer Parameters: {last_layer_params}')

        for parameter in model.sphere_embedding.parameters():
            parameter.requires_grad = False
            
        for block in model.blocks:
            for parameter in block.parameters():
                parameter.requires_grad = False

        for parameter in model.norm.parameters():
            parameter.requires_grad = False

        for parameter in model.edge_degree_embedding.parameters():
            parameter.requires_grad = False
        
        for parameter in model.energy_block.so3_linear_1.parameters():
            parameter.requires_grad = False

        for parameter in model.energy_block.gating_linear.parameters():
            parameter.requires_grad = False
        
        # for parameter in model.energy_block.parameters():
        #     parameter.requires_grad = False

        
        la = Laplace(
            model,
            'regression', 
            hessian_structure='full', 
            subset_of_weights='all', 
            backend=CurvlinopsEF
            )
        la.fit(
            train_loader, 
            is_equiformer=True, 
            target=args.target,
            progress_bar=True
            )
        la.optimize_prior_precision_base(
            method='marglik',
            pred_type='nn',
            val_loader=val_loader,
            loss=criterion,
            verbose=True,
            progress_bar=True
            )

        log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)

        for epoch in range(args.epochs_bayes):
            epoch_start_time = time.perf_counter()

            hyper_optimizer.zero_grad()

            neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()

            info_str = 'Epoch: [{epoch}] Target: [{target}] '.format(epoch=epoch, target=args.target)
            info_str += f'Negative Marginal Likelihood: {neg_marglik:.5f}, '
            info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
            _log.info(info_str)
            wandb.log({"Negative Marginal Log-likelihood": neg_marglik})
            if epoch % 10 == 0:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': la.state_dict(),
                        'optimizer_state_dict': hyper_optimizer.state_dict(),
                        'loss': neg_marglik,
                        }, os.path.join(os.getcwd(), filename(bayes='laplace')))
        y = torch.cat([train_dataset.get(i).y for i in range(len(train_dataset))], dim=0)
        y = y[:, args.target]
        pred_mu, pred_var = la(train_dataset.data)
        utils.calibration_plot(y, None, pred_mu, pred_var, wandb=run)

    if args.variational is True:
        y = torch.cat([train_dataset.get(i).y for i in range(len(train_dataset))], dim=0)
        y = y[:, args.target]

        _, pred_mu, pred_var = model(train_dataset.data)
        utils.calibration_plot(y, None, pred_mu, pred_var, wandb=run)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)