
from models import main_model
import A_dataset
from torch import optim
import time
import numpy as np
import torch

def diffusion_train(configs):
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model = main_model.FGTI(configs).to(configs.device)

    model_optim = optim.Adam(model.parameters(), lr=configs.learning_rate_diff, weight_decay=1e-6)
    p1 = int(0.75 * configs.epoch_diff)
    p2 = int(0.9 * configs.epoch_diff)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optim, milestones=[p1, p2], gamma=0.1
    )

    model.train()
    for epoch in range(configs.epoch_diff):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (observed_data, observed_dataf, observed_mask, observed_tp, gt_mask) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            
            loss = model(observed_data, observed_dataf, observed_mask, observed_tp, gt_mask)
        
            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())
        lr_scheduler.step()
        
        if epoch % 50 == 0 or epoch == configs.epoch_diff-1:
            train_loss = np.average(train_loss)
            print("Epoch: {}. Cost time: {}. Train_loss: {}.".format(epoch + 1, time.time() - epoch_time, train_loss))
    return model

def diffusion_test(configs, model):
    train_loader, test_loader = A_dataset.get_dataset(configs)
    model.eval()

    error_sum = 0
    missing_sum = 0

    target_2d = []
    forecast_2d = []
    eval_p_2d = []
    all_target = []
    all_observed_point = []
    all_observed_time = []
    all_evalpoint = []
    all_generated_samples = []

    print("Testset sum: ", len(test_loader.dataset) // configs.batch + 1)
    
    generate_data2d = []
    start = time.time()
    for i, (observed_data, observed_dataf, observed_mask, observed_tp, gt_mask) in enumerate(test_loader):
        output = model.evaluate(observed_data, observed_dataf, observed_mask, observed_tp, gt_mask)
        imputed_samples, c_target, eval_points, observed_points, observed_time = output

        imputed_samples = imputed_samples.permute(0, 1, 3, 2)  
        c_target = c_target.permute(0, 2, 1)  # (B,L,K)
        eval_points = eval_points.permute(0, 2, 1)
        observed_points = observed_points.permute(0, 2, 1)
        
        #for CRPS
        all_target.append(c_target)  
        all_evalpoint.append(eval_points) 
        all_observed_point.append(observed_points)
        all_observed_time.append(observed_time)  
        all_generated_samples.append(imputed_samples) 

        imputed_sample = imputed_samples.median(dim=1).values.detach().to("cpu")
        imputed_data = observed_mask * observed_data + (1-observed_mask) * imputed_sample
        evalmask = gt_mask - observed_mask

        truth = observed_data * evalmask
        predict = imputed_data * evalmask
        error = torch.sum((truth-predict)**2)
        error_sum += error
        missing_sum += torch.sum(evalmask)
        
        B, L, K = imputed_data.shape
        temp = imputed_data.reshape(B*L, K).detach().to("cpu").numpy()
        generate_data2d.append(temp)

        target_2d.append(observed_data)
        forecast_2d.append(imputed_data)
        eval_p_2d.append(evalmask)

        end = time.time()
        print("time cost for one batch:",end-start)
        start = time.time()

    generate_data2d = np.vstack(generate_data2d)
    np.savetxt("FGTI_Imputation.csv", generate_data2d, delimiter=",")

    target_2d = torch.cat(target_2d, dim=0)
    forecast_2d = torch.cat(forecast_2d, dim=0)
    eval_p_2d = torch.cat(eval_p_2d, dim=0)
    all_target = torch.cat(all_target, dim=0)
    all_evalpoint = torch.cat(all_evalpoint, dim=0)
    all_observed_point = torch.cat(all_observed_point, dim=0)
    all_observed_time = torch.cat(all_observed_time, dim=0)
    all_generated_samples = torch.cat(all_generated_samples, dim=0)


    RMSE = calc_RMSE(target_2d, forecast_2d, eval_p_2d)
    MAE = calc_MAE(target_2d, forecast_2d, eval_p_2d)

    print("RMSE: ", RMSE)
    print("MAE: ", MAE)

def calc_RMSE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    error_mean = torch.mean((target[eval_p] - forecast[eval_p])**2)
    return torch.sqrt(error_mean)

def calc_MAE(target, forecast, eval_points):
    eval_p = torch.where(eval_points == 1)
    error_mean = torch.mean(torch.abs(target[eval_p] - forecast[eval_p]))
    return error_mean
