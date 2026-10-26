@echo off
 echo ========================================
 echo Starting Conventional experiment
 echo ========================================

 python main.py ^
 --pde guckenheimer_dirichlet_2d ^
 --model_name PINN ^
 --layers 3 200 200 200 200 200 1 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini 1.0 ^
 --h_x 0.05 ^
 --h_y 0.05 ^
 --h_t 0.05 ^
 --eval_h_x_learning 0.05 ^
 --eval_h_y_learning 0.05 ^
 --eval_h_t_learning 0.05 ^
 --eval_h_x 0.03125 ^
 --eval_h_y 0.03125 ^
 --eval_h_t 0.03125 ^
 --activation Tanh ^
 --optimizer adamw ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/gkhm2d/results_conventional(gkhm2d_dirichlet)"

 if errorlevel 1 (
    echo Conventional experiment failed.
    exit /b 1
 )

 echo ========================================
 echo Conventional experiments finished
 echo ========================================
 echo ========================================
 echo Starting RBA experiment
 echo ========================================

 python main.py ^
 --pde guckenheimer_dirichlet_2d ^
 --model_name PINN ^
 --layers 3 200 200 200 200 200 1 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini 1.0 ^
 --h_x 0.05 ^
 --h_y 0.05 ^
 --h_t 0.05 ^
 --eval_h_x_learning 0.05 ^
 --eval_h_y_learning 0.05 ^
 --eval_h_t_learning 0.05 ^
 --eval_h_x 0.03125 ^
 --eval_h_y 0.03125 ^
 --eval_h_t 0.03125 ^
 --activation Tanh ^
 --use_rba ^
 --scheduler rba ^
 --use_weight_norm ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --optimizer adamW ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/gkhm2d/results_rba(gkhm2d_dirichlet)"

 if errorlevel 1 (
     echo RBA experiment failed.
     exit /b 1
 )

 echo ========================================
 echo RBA experiments finished
 echo ========================================
 echo ========================================
 echo Starting NTK experiment
 echo ========================================

 python main.py ^
 --pde guckenheimer_dirichlet_2d ^
 --model_name PINN ^
 --layers 3 200 200 200 200 200 1 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini 1.0 ^
 --h_x 0.05 ^
 --h_y 0.05 ^
 --h_t 0.05 ^
 --eval_h_x_learning 0.05 ^
 --eval_h_y_learning 0.05 ^
 --eval_h_t_learning 0.05 ^
 --eval_h_x 0.03125 ^
 --eval_h_y 0.03125 ^
 --eval_h_t 0.03125 ^
 --activation Tanh ^
 --use_ntk ^
 --ntk_update_every 1000 ^
 --ntk_probes 0 ^
 --optimizer adamw ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/gkhm2d/results_NTK(gkhm2d_dirichlet)"

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
 --pde guckenheimer_dirichlet_2d ^
 --model_name PINN ^
 --layers 3 200 200 200 200 200 1 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini 1.0 ^
 --h_x 0.05 ^
 --h_y 0.05 ^
 --h_t 0.05 ^
 --eval_h_x_learning 0.05 ^
 --eval_h_y_learning 0.05 ^
 --eval_h_t_learning 0.05 ^
 --eval_h_x 0.03125 ^
 --eval_h_y 0.03125 ^
 --eval_h_t 0.03125 ^
 --activation Tanh ^
 --use_asa ^
 --sa_init_pde 10 ^
 --sa_init_bc 1 ^
 --sa_init_bc_grad 1 ^
 --sa_init_ini 1 ^
 --sa_weight_mode square ^
 --sa_optimizer adamw ^
 --ntk_update_every 1000 ^
 --ntk_probes 0 ^
 --sa_lr 1e-4 ^
 --optimizer adamw ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/gkhm2d/results_aSA(gkhm2d_dirichlet)"

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
 --pde guckenheimer_dirichlet_2d ^
 --model_name PINN ^
 --layers 3 200 200 200 200 200 1 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini 1.0 ^
 --h_x 0.05 ^
 --h_y 0.05 ^
 --h_t 0.05 ^
 --eval_h_x_learning 0.05 ^
 --eval_h_y_learning 0.05 ^
 --eval_h_t_learning 0.05 ^
 --eval_h_x 0.03125 ^
 --eval_h_y 0.03125 ^
 --eval_h_t 0.03125 ^
 --activation Tanh ^
 --use_sa ^
 --sa_init_pde 10 ^
 --sa_init_bc 1 ^
 --sa_init_bc_grad 1 ^
 --sa_init_ini 1 ^
 --sa_weight_mode square ^
 --sa_optimizer adamw ^
 --sa_lr 1e-4 ^
 --optimizer adamw ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/gkhm2d/results_SA(gkhm2d_dirichlet)"

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
--pde guckenheimer_dirichlet_2d ^
--model_name PINN ^
--layers 3 200 200 200 200 200 1 ^
--lambda_pde 1.0 ^
--lambda_bc 1.0 ^
--lambda_ini 1.0 ^
--h_x 0.05 ^
--h_y 0.05 ^
--h_t 0.05 ^
--eval_h_x_learning 0.05 ^
--eval_h_y_learning 0.05 ^
--eval_h_t_learning 0.05 ^
--eval_h_x 0.03125 ^
--eval_h_y 0.03125 ^
--eval_h_t 0.03125 ^
--activation Tanh ^
--use_causal ^
--causal_epsilon_schedule 10 ^
--causal_stage_epochs 20000 ^
--causal_delta 0.99 ^
--causal_normalization separate ^
--no-causal_include_initial ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--optimizer adamW ^
--num_epochs 20000 ^
--lbfgs_epochs 200 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/gkhm2d/results_causal(gkhm2d_dirichlet)"

if errorlevel 1 (
   echo Causal experiment failed.
   exit /b 1
)

echo ========================================
echo Causal experiments finished
echo ========================================
 echo ========================================
 echo Starting BRDR experiment
 echo ========================================

 python main.py ^
 --pde guckenheimer_dirichlet_2d ^
 --model_name PINN ^
 --layers 3 200 200 200 200 200 1 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini 1.0 ^
 --h_x 0.05 ^
 --h_y 0.05 ^
 --h_t 0.05 ^
 --eval_h_x_learning 0.05 ^
 --eval_h_y_learning 0.05 ^
 --eval_h_t_learning 0.05 ^
 --eval_h_x 0.03125 ^
 --eval_h_y 0.03125 ^
 --eval_h_t 0.03125 ^
 --activation Tanh ^
 --use_brdr ^
 --scheduler brdr ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --optimizer adamW ^
 --use_weight_norm ^
 --num_epochs 20000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/gkhm2d/results_brdr(gkhm2d_dirichlet)"

 if errorlevel 1 (
    echo BRDR experiment failed.
    exit /b 1
 )
 echo ========================================
 echo BRDR experiments finished.
 echo ========================================
echo ========================================
echo All experiments finished
echo ========================================

pause
