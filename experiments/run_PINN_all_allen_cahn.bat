@echo off
echo ========================================
echo Starting Conventional experiment
echo ========================================

python main.py ^
--pde allen_cahn ^
--model_name PINNFourier ^
--layers 128 128 128 128 128 128 1 ^
--input_dim 2 ^
--embed_dim 64 ^
--lambda_pde 1.0 ^
--lambda_bc 10.0 ^
--lambda_bc_grad 10.0 ^
--lambda_ini 100.0 ^
--h_x 0.0625 ^
--h_t 0.015625 ^
--lr 1e-3 ^
--num_epochs 100000 ^
--lbfgs_epochs 1000 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/allen_cahn/results_conventional(allen_cahn)"

if errorlevel 1 (
    echo Conventional experiment failed.
    exit /b 1
)

echo ========================================
echo Conventional experiments finished
echo ========================================
 echo ========================================
 echo Starting NTK experiment
 echo ========================================

 python main.py ^
 --pde allen_cahn ^
 --model_name PINNFourier ^
 --layers 128 128 128 128 128 128 1 ^
 --input_dim 2 ^
 --embed_dim 64 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_bc_grad 1.0 ^
 --lambda_ini 1.0 ^
 --h_x 0.0625 ^
 --h_t 0.015625 ^
 --activation Tanh ^
 --use_ntk ^
 --optimizer adam ^
 --lr 1e-3 ^
 --num_epochs 100000 ^
 --lbfgs_epochs 1000 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/allen_cahn/results_NTK(allen_cahn)"

 if errorlevel 1 (
    echo NTK experiment failed.
    exit /b 1
 )

 echo ========================================
 echo NTK experiments finished
 echo ========================================
 echo ========================================
 echo Starting aSA experiment
 echo ========================================

 python main.py ^
 --pde allen_cahn ^
 --model_name PINNFourier ^
 --layers 128 128 128 128 128 128 1 ^
 --input_dim 2 ^
 --embed_dim 64 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_bc_grad 1.0 ^
 --lambda_ini 1.0 ^
 --h_x 0.0625 ^
 --h_t 0.015625 ^
 --activation Tanh ^
 --use_asa ^
 --sa_init_pde 1 ^
 --sa_init_bc 3 ^
 --sa_init_bc_grad 3 ^
 --sa_init_ini 10 ^
 --sa_weight_mode linear ^
 --sa_optimizer sgd ^
 --ntk_update_every 1000 ^
 --ntk_probes 0 ^
 --sa_lr 1e-3 ^
 --optimizer adam ^
 --lr 1e-3 ^
 --num_epochs 100000 ^
 --lbfgs_epochs 1000 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/allen_cahn/results_aSA(allen_cahn)"

 if errorlevel 1 (
    echo aSA experiment failed.
    exit /b 1
 )

 echo ========================================
 echo aSA experiments finished
 echo ========================================
 echo ========================================
 echo Starting SA experiment
 echo ========================================

 python main.py ^
 --pde allen_cahn ^
 --model_name PINNFourier ^
 --layers 128 128 128 128 128 128 1 ^
 --input_dim 2 ^
 --embed_dim 64 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_bc_grad 1.0 ^
 --lambda_ini 1.0 ^
 --h_x 0.0625 ^
 --h_t 0.015625 ^
 --activation Tanh ^
 --use_sa ^
 --sa_init_pde 1 ^
 --sa_init_bc 3 ^
 --sa_init_bc_grad 3 ^
 --sa_init_ini 10 ^
 --sa_weight_mode linear ^
 --sa_optimizer sgd ^
 --sa_lr 1e-3 ^
 --optimizer adam ^
 --lr 1e-3 ^
 --num_epochs 100000 ^
 --lbfgs_epochs 1000 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/allen_cahn/results_SA(allen_cahn)"

 if errorlevel 1 (
    echo SA experiment failed.
    exit /b 1
 )

 echo ========================================
 echo SA experiments finished
 echo ========================================
echo ========================================
echo Starting Causal experiment
echo ========================================

python main.py ^
--pde allen_cahn ^
--model_name PINNFourier ^
--layers 128 128 128 128 128 128 1 ^
--input_dim 2 ^
--embed_dim 64 ^
--use_causal ^
--causal_epsilon_schedule 10 ^
--causal_stage_epochs 100000 ^
--causal_delta 0.99 ^
--causal_normalization separate ^
--no-causal_include_initial ^
--lambda_pde 1.0 ^
--lambda_bc 10.0 ^
--lambda_bc_grad 10.0 ^
--lambda_ini 100.0 ^
--h_x 0.0625 ^
--h_t 0.015625 ^
--lr 1e-3 ^
--activation Tanh ^
--optimizer Adam ^
--num_epochs 100000 ^
--lbfgs_epochs 1000 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/allen_cahn/results_causal(allen_cahn)"

if errorlevel 1 (
   echo Causal experiment failed.
   exit /b 1
)

echo ========================================
echo Causal experiments finished
echo ========================================
echo ========================================
echo Starting RBA experiment
echo ========================================

python main.py ^
--model_name PINNFourier ^
--use_rba ^
--scheduler rba ^
--layers 128 128 128 128 128 128 1 ^
--input_dim 2 ^
--embed_dim 64 ^
--h_x 0.0625 ^
--h_t 0.015625 ^
--lambda_pde 1.0 ^
--lambda_ini 100.0 ^
--lambda_bc 10.0 ^
--lambda_bc_grad 10.0 ^
--lr 1e-3 ^
--use_weight_norm ^
--activation Tanh ^
--optimizer Adam ^
--num_epochs 100000 ^
 --lbfgs_epochs 1000 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/allen_cahn/results_rba(allen_cahn)"

if errorlevel 1 (
    echo RBA experiment failed.
    exit /b 1
)

echo ========================================
echo RBA experiments finished
echo ========================================
 echo ========================================
 echo Starting BRDR experiment
 echo ========================================

 python main.py ^
 --model_name PINNFourier ^
 --use_brdr ^
 --scheduler brdr ^
 --layers 128 128 128 128 128 128 1 ^
 --input_dim 2 ^
 --embed_dim 64 ^
 --h_x 0.0625 ^
 --h_t 0.015625 ^
 --lambda_pde 1.0 ^
 --lambda_bc 10.0 ^
 --lambda_bc_grad 10.0 ^
 --lambda_ini 100.0 ^
 --lr 1e-3 ^
 --use_weight_norm ^
 --activation Tanh ^
 --optimizer Adam ^
 --num_epochs 100000 ^
 --lbfgs_epochs 1000 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/allen_cahn/results_brdr(allen_cahn)"

 if errorlevel 1 (
     echo BRDR experiment failed.
     exit /b 1
 )
 echo ========================================
 echo BRDR experiments finished
 echo ========================================
echo ========================================
echo All experiments finished
echo ========================================

pause
