@echo off
 echo ========================================
 echo Starting Conventional experiment
 echo ========================================

 python main.py ^
 --pde burgers_1d ^
 --model_name PINN ^
 --layers 2 50 50 50 50 50 1 ^
 --lambda_pde 1.0 ^
 --lambda_bc 5.0 ^
 --lambda_ini1 5.0 ^
 --lambda_ini2 5.0 ^
 --h_x 0.015625 ^
 --h_t 0.015625 ^
 --lr 1e-5 ^
 --weight_decay 1e-3 ^
 --optimizer adamw ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers/results_conventional(burgers)"

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
 --pde burgers_1d ^
 --model_name PINN ^
 --layers 2 50 50 50 50 50 1 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini1 1.0 ^
 --lambda_ini2 1.0 ^
 --h_x 0.015625 ^
 --h_t 0.015625 ^
 --lr 1e-5 ^
 --weight_decay 1e-3 ^
 --optimizer adamw ^
 --activation Tanh ^
 --use_ntk ^
 --ntk_update_every 2500 ^
 --ntk_probes 0 ^
 --optimizer adamw ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers/results_NTK(burgers)"

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
 --pde burgers_1d ^
 --model_name PINN ^
 --layers 2 50 50 50 50 50 1 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini1 1.0 ^
 --lambda_ini2 1.0 ^
 --h_x 0.015625 ^
 --h_t 0.015625 ^
 --activation Tanh ^
 --use_asa ^
 --sa_init_pde 10 ^
 --sa_init_bc 1 ^
 --sa_init_ini 1 ^
 --sa_weight_mode square ^
 --sa_optimizer adamw ^
 --ntk_update_every 2500 ^
 --ntk_probes 0 ^
 --sa_lr 1e-4 ^
 --optimizer adamw ^
 --lr 1e-5 ^
 --weight_decay 1e-3 ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers/results_aSA(burgers)"

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
 --pde burgers_1d ^
 --model_name PINN ^
 --layers 2 50 50 50 50 50 1 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini1 1.0 ^
 --lambda_ini2 1.0 ^
 --h_x 0.015625 ^
 --h_t 0.015625 ^
 --activation Tanh ^
 --use_sa ^
 --sa_init_pde 10 ^
 --sa_init_bc 1 ^
 --sa_init_ini 1 ^
 --sa_weight_mode square ^
 --sa_optimizer adamw ^
 --sa_lr 1e-4 ^
 --optimizer adamw ^
 --lr 1e-5 ^
 --weight_decay 1e-3 ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers/results_SA(burgers)"

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
--pde burgers_1d ^
--model_name PINN ^
--layers 2 50 50 50 50 50 1 ^
--use_causal ^
--causal_epsilon_schedule 1 ^
--causal_stage_epochs 20000 ^
--causal_delta 0.99 ^
--causal_normalization separate ^
--no-causal_include_initial ^
--lambda_pde 1.0 ^
--lambda_bc 5.0 ^
--lambda_ini1 5.0 ^
--lambda_ini2 5.0 ^
--h_x 0.015625 ^
--h_t 0.015625 ^
--lr 1e-5 ^
--weight_decay 1e-3 ^
--optimizer adamw ^
--num_epochs 20000 ^
--lbfgs_epochs 200 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/burgers/results_causal(burgers)"

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
 --model_name PINN ^
 --pde burgers_1d ^
 --use_rba ^
 --scheduler rba ^
 --layers 2 50 50 50 50 50 1 ^
 --h_x 0.015625 ^
 --h_t 0.015625 ^
 --lambda_pde 1.0 ^
 --lambda_bc 5.0 ^
 --lambda_ini1 5.0 ^
 --lambda_ini2 5.0 ^
 --use_weight_norm ^
 --activation Tanh ^
 --lr 1e-5 ^
 --weight_decay 1e-3 ^
 --optimizer adamw ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers/results_rba(burgers)"

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
 --model_name PINN ^
 --pde burgers_1d ^
 --use_brdr ^
 --scheduler brdr ^
 --layers 2 50 50 50 50 50 1 ^
 --h_x 0.015625 ^
 --h_t 0.015625 ^
 --lambda_pde 1.0 ^
 --lambda_bc 5.0 ^
 --lambda_ini1 5.0 ^
 --lambda_ini2 5.0 ^
 --lr 1e-5 ^
 --weight_decay 1e-3 ^
 --optimizer adamw ^
 --use_weight_norm ^
 --activation Tanh ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers/results_brdr(burgers)"

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
