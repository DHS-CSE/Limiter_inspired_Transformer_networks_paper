@echo off

echo ========================================
echo Starting Multi experiment
echo ========================================

python main.py ^
--pde burgers_1d ^
--model_name BurgersSLFormerFixedSharpAnalysis ^
--fourier_scale 1.0 ^
--embed_dim 64 ^
--num_heads 1 ^
--depth 2 ^
--h_x 0.015625 ^
--h_t 0.015625 ^
--eval_h_x 0.01 ^
--eval_h_t 0.01 ^
--lambda_pde 1.0 ^
--lambda_ini 5.0 ^
--lambda_bc 5.0 ^
--lr 1e-5 ^
--weight_decay 1e-3 ^
--optimizer adamW ^
--sharp_const 1.0 ^
--num_epochs 15000 ^
--lbfgs_epochs 200 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/burgers/results_multi(burgers)"

if errorlevel 1 (
    echo Multi experiment failed.
    exit /b 1
)

echo ========================================
echo Multi experiments finished
echo ========================================
echo ========================================
echo Starting Multi NoSlope experiment
echo ========================================

python main.py ^
--pde burgers_1d ^
--model_name BurgersSLFormerFixedSharpAnalysis_NoSlope ^
--fourier_scale 1.0 ^
--embed_dim 64 ^
--num_heads 1 ^
--depth 2 ^
--h_x 0.015625 ^
--h_t 0.015625 ^
--eval_h_x 0.01 ^
--eval_h_t 0.01 ^
--lambda_pde 1.0 ^
--lambda_ini 5.0 ^
--lambda_bc 5.0 ^
--lr 1e-5 ^
--weight_decay 1e-3 ^
--optimizer adamW ^
--sharp_const 1.0 ^
--num_epochs 15000 ^
--lbfgs_epochs 200 ^
--seeds 1 ^
--save_dir "runs/burgers/results_multi_NoSlope(burgers)"

if errorlevel 1 (
    echo Multi NoSlope experiment failed.
    exit /b 1
)

echo ========================================
echo Multi NoSlope experiments finished
echo ========================================
 echo ========================================
 echo Starting Single experiment
 echo ========================================

 python main.py ^
 --pde burgers_1d ^
 --model_name BurgersSLFormerFixedSharpAnalysis_Single ^
 --fourier_scale 1.0 ^
 --embed_dim 96 ^
 --num_heads 1 ^
 --depth 2 ^
 --h_x 0.015625 ^
 --h_t 0.015625 ^
 --eval_h_x 0.01 ^
 --eval_h_t 0.01 ^
 --lambda_pde 1.0 ^
 --lambda_ini 5.0 ^
 --lambda_bc 5.0 ^
 --lr 1e-5 ^
 --weight_decay 1e-3 ^
 --optimizer adamW ^
 --sharp_const 1.0 ^
 --num_epochs 15000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers/results_single(burgers)"

 if errorlevel 1 (
     echo Single experiment failed.
     exit /b 1
 )

 echo ========================================
 echo Single experiments finished
 echo ========================================
 echo ========================================
 echo Starting Single NoSlope experiment
 echo ========================================

 python main.py ^
 --pde burgers_1d ^
 --model_name BurgersSLFormer_Single_NoSlope ^
 --fourier_scale 1.0 ^
 --embed_dim 96 ^
 --num_heads 1 ^
 --depth 2 ^
 --h_x 0.015625 ^
 --h_t 0.015625 ^
 --eval_h_x 0.01 ^
 --eval_h_t 0.01 ^
 --lambda_pde 1.0 ^
 --lambda_ini 5.0 ^
 --lambda_bc 5.0 ^
 --lr 1e-5 ^
 --weight_decay 1e-3 ^
 --optimizer adamW ^
 --sharp_const 1.0 ^
 --num_epochs 15000 ^
 --lbfgs_epochs 200 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers/results_single_NoSlope(burgers)"

 if errorlevel 1 (
     echo Single experiment failed.
     exit /b 1
 )

 echo ========================================
 echo Single NoSlope experiments finished
 echo ========================================
echo ========================================
echo Starting Multi experiment
echo ========================================

python main.py ^
--pde burgers_1d ^
--model_name BurgersSLFormerFixedSharpAnalysis_Superbee ^
--fourier_scale 1.0 ^
--embed_dim 64 ^
--num_heads 1 ^
--depth 2 ^
--h_x 0.015625 ^
--h_t 0.015625 ^
--eval_h_x 0.01 ^
--eval_h_t 0.01 ^
--lambda_pde 1.0 ^
--lambda_ini 5.0 ^
--lambda_bc 5.0 ^
--lr 1e-5 ^
--weight_decay 1e-3 ^
--optimizer adamW ^
--sharp_const 1.0 ^
--num_epochs 15000 ^
--lbfgs_epochs 200 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/burgers/results_multi_superbee(burgers)"

if errorlevel 1 (
    echo Multi Superbee experiment failed.
    exit /b 1
)

echo ========================================
echo Multi Superbee experiments finished
echo ========================================
echo ========================================
echo Starting Multi experiment
echo ========================================

python main.py ^
--pde burgers_1d ^
--model_name BurgersSLFormerFixedSharpAnalysis_MC ^
--fourier_scale 1.0 ^
--embed_dim 64 ^
--num_heads 1 ^
--depth 2 ^
--h_x 0.015625 ^
--h_t 0.015625 ^
--eval_h_x 0.01 ^
--eval_h_t 0.01 ^
--lambda_pde 1.0 ^
--lambda_ini 5.0 ^
--lambda_bc 5.0 ^
--lr 1e-5 ^
--weight_decay 1e-3 ^
--optimizer adamW ^
--sharp_const 1.0 ^
--num_epochs 15000 ^
--lbfgs_epochs 200 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/burgers/results_multi_MC(burgers)"

if errorlevel 1 (
    echo Multi MC experiment failed.
    exit /b 1
)

echo ========================================
echo Multi MC experiments finished
echo ========================================
echo ========================================
echo Starting Multi experiment
echo ========================================

python main.py ^
--pde burgers_1d ^
--model_name BurgersSLFormerFixedSharpAnalysis_Random ^
--fourier_scale 1.0 ^
--embed_dim 64 ^
--num_heads 1 ^
--depth 2 ^
--point_type random ^
--h_x 0.015625 ^
--h_t 0.015625 ^
--n_domain_x 127 ^
--eval_h_x_learning 0.015625 ^
--eval_h_t_learning 0.015625 ^
--eval_h_x 0.01 ^
--eval_h_t 0.01 ^
--lambda_pde 1.0 ^
--lambda_ini 5.0 ^
--lambda_bc 5.0 ^
--lr 1e-5 ^
--weight_decay 1e-3 ^
--optimizer adamW ^
--sharp_const 1.0 ^
--num_epochs 15000 ^
--lbfgs_epochs 200 ^
--seeds 1 2 3 4 5 ^
--save_dir "runs/burgers/results_multi_random(burgers)"

if errorlevel 1 (
    echo Multi experiment failed.
    exit /b 1
)

echo ========================================
echo Multi experiments finished
echo ========================================
echo ========================================
echo ALL experiments finished
echo ========================================



pause
