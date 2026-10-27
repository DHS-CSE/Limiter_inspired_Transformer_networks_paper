@echo off
 echo ========================================
 echo Starting Multi experiment
 echo ========================================

 python main.py ^
 --pde guckenheimer_dirichlet_2d ^
 --model_name Burgers2DSLFormerFixedSharpAnalysis ^
 --fourier_scale 0.25 ^
 --embed_dim 64 ^
 --num_heads 1 ^
 --depth 2 ^
 --activation Tanh ^
 --h_x 0.05 ^
 --h_y 0.05 ^
 --h_t 0.05 ^
 --eval_h_x_learning 0.05 ^
 --eval_h_y_learning 0.05 ^
 --eval_h_t_learning 0.05 ^
 --eval_h_x 0.03125 ^
 --eval_h_y 0.03125 ^
 --eval_h_t 0.03125 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini 1.0 ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --optimizer adamw ^
 --sharp_const 1.0 ^
 --num_epochs 5000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/gkhm2d/results_multi(gkhm2d_dirichlet)"

 if errorlevel 1 (
     echo Multi experiment failed.
     exit /b 1
 )

 echo ========================================
 echo Multi experiments finished
 echo ========================================
 echo ========================================
 echo Starting Single experiment
 echo ========================================

 python main.py ^
 --pde guckenheimer_dirichlet_2d ^
 --model_name Burgers2DSLSingleFormerAnalysis ^
 --fourier_scale 0.25 ^
 --embed_dim 64 ^
 --num_heads 1 ^
 --depth 2 ^
 --activation Tanh ^
 --h_x 0.05 ^
 --h_y 0.05 ^
 --h_t 0.05 ^
 --eval_h_x_learning 0.05 ^
 --eval_h_y_learning 0.05 ^
 --eval_h_t_learning 0.05 ^
 --eval_h_x 0.03125 ^
 --eval_h_y 0.03125 ^
 --eval_h_t 0.03125 ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini 1.0 ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --optimizer adamW ^
 --sharp_const 1.0 ^
 --num_epochs 5000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/gkhm2d/results_single(gkhm2d_dirichlet)"

 if errorlevel 1 (
     echo Single experiment failed.
     exit /b 1
 )

 echo ========================================
 echo Single experiments finished
 echo ========================================
echo ========================================
echo Starting Multi NoSlope experiment
echo ========================================

python main.py ^
--pde guckenheimer_dirichlet_2d ^
--model_name Burgers2DSLFormer_NoSlope ^
--fourier_scale 0.25 ^
--embed_dim 64 ^
--num_heads 1 ^
--depth 2 ^
--activation Tanh ^
--h_x 0.05 ^
--h_y 0.05 ^
--h_t 0.05 ^
--eval_h_x_learning 0.05 ^
--eval_h_y_learning 0.05 ^
--eval_h_t_learning 0.05 ^
--eval_h_x 0.03125 ^
--eval_h_y 0.03125 ^
--eval_h_t 0.03125 ^
--lambda_pde 1.0 ^
--lambda_bc 1.0 ^
--lambda_ini 1.0 ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--optimizer adamW ^
--sharp_const 1.0 ^
--num_epochs 5000 ^
--lbfgs_epochs 200 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/gkhm2d/results_multi_NoSlope(gkhm2d_dirichlet)"

if errorlevel 1 (
    echo Multi NoSlope experiment failed.
    exit /b 1
)

echo ========================================
echo Multi NoSlope experiments finished
echo ========================================
echo ========================================
echo Starting Single NoSlope experiment
echo ========================================

python main.py ^
--pde guckenheimer_dirichlet_2d ^
--model_name Burgers2DSLSingleFormer_NoSlope ^
--fourier_scale 0.25 ^
--embed_dim 64 ^
--num_heads 1 ^
--depth 2 ^
--activation Tanh ^
--h_x 0.05 ^
--h_y 0.05 ^
--h_t 0.05 ^
--eval_h_x_learning 0.05 ^
--eval_h_y_learning 0.05 ^
--eval_h_t_learning 0.05 ^
--eval_h_x 0.03125 ^
--eval_h_y 0.03125 ^
--eval_h_t 0.03125 ^
--lambda_pde 1.0 ^
--lambda_bc 1.0 ^
--lambda_ini 1.0 ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--optimizer adamW ^
--sharp_const 1.0 ^
--num_epochs 5000 ^
--lbfgs_epochs 200 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/gkhm2d/results_single_NoSlope(gkhm2d_dirichlet)"

if errorlevel 1 (
    echo Single NoSlope experiment failed.
    exit /b 1
)

echo ========================================
echo Single NoSlope experiments finished
echo ========================================
echo ========================================
echo All experiments finished
echo ========================================
pause
