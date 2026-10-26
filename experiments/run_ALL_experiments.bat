@echo off

echo ========================================
echo Starting Single experiment
echo ========================================

python main.py ^
--model_name SLSingleFormerFixedSharpAnalysis ^
--embed_dim 96 ^
--num_heads 2 ^
--depth 2 ^
--lambda_pde 1.0 ^
--lambda_ini 100.0 ^
--lambda_bc 10.0 ^
--lambda_bc_grad 10.0 ^
--lr 1e-3 ^
--optimizer adam ^
--num_epochs 8000 ^
--lbfgs_epochs 1000 ^
--seeds 1 ^
--analysis_every 100 ^
--save_dir "runs/allen_cahn/results_single(allen_cahn)"

if errorlevel 1 (
   echo Single experiment failed.
   exit /b 1
)

echo ========================================
echo Single experiments finished
echo ========================================
echo ========================================
echo Starting Multi experiment
echo ========================================

python main.py ^
--model_name SLFormerFixedSharpAnalysis ^
--embed_dim 64 ^
--num_heads 2 ^
--depth 2 ^
--lambda_pde 1.0 ^
--lambda_ini 100.0 ^
--lambda_bc 10.0 ^
--lambda_bc_grad 10.0 ^
--lr 1e-3 ^
--optimizer Adam ^
--num_epochs 8000 ^
--lbfgs_epochs 1000 ^
--analysis_every 100 ^
--seeds 1 ^
--save_dir "runs/allen_cahn/results_multi(allen_cahn)"

if errorlevel 1 (
   echo Multi experiment failed.
   exit /b 1
)

echo ========================================
echo Multi experiments finished
echo ========================================
 echo ========================================
 echo Starting Single NoSlope experiment
 echo ========================================

 python main.py ^
 --model_name SLSingleFormerFixedSharpAnalysis_NoSlope ^
 --embed_dim 96 ^
 --num_heads 2 ^
 --depth 2 ^
 --lambda_pde 1.0 ^
 --lambda_ini 100.0 ^
 --lambda_bc 10.0 ^
 --lambda_bc_grad 10.0 ^
 --lr 1e-3 ^
 --optimizer adam ^
 --num_epochs 8000 ^
 --lbfgs_epochs 1000 ^
 --seeds 1 2 3 4 5 ^
 --analysis_every 100 ^
 --save_dir "runs/allen_cahn/results_single_NoSlope(allen_cahn)"

 if errorlevel 1 (
    echo Single NoSlope experiment failed.
    exit /b 1
 )

 echo ========================================
 echo Single NoSlope experiments finished
 echo ========================================
 echo ========================================
 echo Starting Multi NoSlope experiment
 echo ========================================

 python main.py ^
 --model_name SLFormerFixedSharpAnalysis_NoSlope ^
 --embed_dim 64 ^
 --num_heads 2 ^
 --depth 2 ^
 --lambda_pde 1.0 ^
 --lambda_ini 100.0 ^
 --lambda_bc 10.0 ^
 --lambda_bc_grad 10.0 ^
 --lr 1e-3 ^
 --optimizer Adam ^
 --num_epochs 8000 ^
 --lbfgs_epochs 1000 ^
 --analysis_every 100 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/allen_cahn/results_multi_NoSlope(allen_cahn)"

 if errorlevel 1 (
    echo Multi NoSlope experiment failed.
    exit /b 1
 )

 echo ========================================
 echo Multi NoSlope experiments finished
 echo ========================================
echo ========================================
echo ALL experiments finished
echo ========================================

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
--seeds 1 ^
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
 --seeds 1 2 3 4 5 ^
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
--seeds 1 ^
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
--seeds 2 3 4 5 ^
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

echo ========================================
echo Starting Multi experiment
echo ========================================

python main.py ^
--pde burgers_2d ^
--model_name Burgers2DSLFormerFixedSharpAnalysis ^
--fourier_scale 0.25 ^
--embed_dim 64 ^
--num_heads 2 ^
--depth 2 ^
--h_x 0.0625 ^
--h_y 0.0625 ^
--h_t 0.0625 ^
--eval_h_x 0.05 ^
--eval_h_y 0.05 ^
--eval_h_t 0.05 ^
--lambda_pde 1.0 ^
--lambda_ini 1.0 ^
--lambda_bc 1.0 ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--optimizer adamW ^
--lambda_pde 1.0 ^
--lambda_ini 1.0 ^
--lambda_bc 1.0 ^
--sharp_const 1.0 ^
--activation SiLU ^
--num_epochs 10000 ^
--lbfgs_epochs 100 ^
--seeds 1 ^
--save_dir "runs/burgers2d/results_multi(burgers2d)"

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
--pde burgers_2d ^
--model_name Burgers2DSLSingleFormerAnalysis ^
--fourier_scale 0.25 ^
--embed_dim 64 ^
--num_heads 2 ^
--depth 2 ^
--h_x 0.0625 ^
--h_y 0.0625 ^
--h_t 0.0625 ^
--eval_h_x 0.05 ^
--eval_h_y 0.05 ^
--eval_h_t 0.05 ^
--lambda_pde 1.0 ^
--lambda_ini 1.0 ^
--lambda_bc 1.0 ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--optimizer adamW ^
--lambda_pde 1.0 ^
--lambda_ini 1.0 ^
--lambda_bc 1.0 ^
--sharp_const 1.0 ^
--activation SiLU ^
--num_epochs 10000 ^
--lbfgs_epochs 100 ^
--seeds 1 ^
--save_dir "runs/burgers2d/results_single(burgers2d)"

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
 --pde burgers_2d ^
 --model_name Burgers2DSLFormer_NoSlope ^
 --fourier_scale 0.25 ^
 --embed_dim 64 ^
 --num_heads 2 ^
 --depth 2 ^
 --h_x 0.0625 ^
 --h_y 0.0625 ^
 --h_t 0.0625 ^
 --eval_h_x 0.05 ^
 --eval_h_y 0.05 ^
 --eval_h_t 0.05 ^
 --lambda_pde 1.0 ^
 --lambda_ini 1.0 ^
 --lambda_bc 1.0 ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --activation SiLU ^
 --optimizer adamW ^
 --sharp_const 1.0 ^
 --num_epochs 10000 ^
 --lbfgs_epochs 100 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers2d/results_multi_NoSlope(burgers2d)"

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
 --pde burgers_2d ^
 --model_name Burgers2DSLSingleFormer_NoSlope ^
 --fourier_scale 0.25 ^
 --embed_dim 64 ^
 --num_heads 2 ^
 --depth 2 ^
 --h_x 0.0625 ^
 --h_y 0.0625 ^
 --h_t 0.0625 ^
 --eval_h_x 0.05 ^
 --eval_h_y 0.05 ^
 --eval_h_t 0.05 ^
 --lambda_pde 1.0 ^
 --lambda_ini 1.0 ^
 --lambda_bc 1.0 ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --optimizer adamW ^
 --sharp_const 1.0 ^
 --activation SiLU ^
 --num_epochs 10000 ^
 --lbfgs_epochs 100 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers2d/results_single_NoSlope(burgers2d)"

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
--activation SiLU ^
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
--seeds 1 ^
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
--activation SiLU ^
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
--seeds 1 ^
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
 --activation SiLU ^
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
 --activation SiLU ^
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
--seeds 1 ^
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
--seeds 1 ^
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
 --seeds 1 ^
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
--seeds 1 ^
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
--seeds 1 ^
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
--seeds 1 ^
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
--seeds 1 ^
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
--seeds 1 ^
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
--seeds 1 ^
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
echo ========================================
echo Starting Conventional experiment
echo ========================================

python main.py ^
--pde burgers_2d ^
--model_name PINN ^
--layers 3 200 200 200 200 200 1 ^
--lambda_pde 1.0 ^
--lambda_bc 1.0 ^
--lambda_ini 1.0 ^
--h_x 0.0625 ^
--h_y 0.0625 ^
--h_t 0.0625 ^
--eval_h_x 0.05 ^
--eval_h_y 0.05 ^
--eval_h_t 0.05 ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--optimizer adamW ^
--num_epochs 10000 ^
--lbfgs_epochs 100 ^
--seeds 1 ^
--save_dir "runs/burgers2d/results_conventional(burgers2d)"

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
--pde burgers_2d ^
--model_name PINN ^
--layers 3 200 200 200 200 200 1 ^
--lambda_pde 1.0 ^
--lambda_bc 1.0 ^
--lambda_ini 1.0 ^
--h_x 0.0625 ^
--h_y 0.0625 ^
--h_t 0.0625 ^
--eval_h_x 0.05 ^
--eval_h_y 0.05 ^
--eval_h_t 0.05 ^
--activation Tanh ^
--use_ntk ^
--ntk_update_every 1000 ^
--ntk_probes 0 ^
--optimizer adamw ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--num_epochs 10000 ^
--lbfgs_epochs 100 ^
--seeds 1 ^
--save_dir "runs/burgers2d/results_NTK(burgers2d)"

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
--pde burgers_2d ^
--model_name PINN ^
--layers 3 200 200 200 200 200 1 ^
--lambda_pde 1.0 ^
--lambda_bc 1.0 ^
--lambda_ini 1.0 ^
--h_x 0.0625 ^
--h_y 0.0625 ^
--h_t 0.0625 ^
--eval_h_x 0.05 ^
--eval_h_y 0.05 ^
--eval_h_t 0.05 ^
--activation Tanh ^
--use_asa ^
--sa_init_pde 1 ^
--sa_init_bc 1 ^
--sa_init_ini 1 ^
--sa_weight_mode square ^
--sa_optimizer adamw ^
--ntk_update_every 1000 ^
--ntk_probes 0 ^
--sa_lr 1e-4 ^
--optimizer adamw ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--num_epochs 10000 ^
--lbfgs_epochs 100 ^
--seeds 1 ^
--save_dir "runs/burgers2d/results_aSA(burgers2d)"

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
--pde burgers_2d ^
--model_name PINN ^
--layers 3 200 200 200 200 200 1 ^
--lambda_pde 1.0 ^
--lambda_bc 1.0 ^
--lambda_ini 1.0 ^
--h_x 0.0625 ^
--h_y 0.0625 ^
--h_t 0.0625 ^
--eval_h_x 0.05 ^
--eval_h_y 0.05 ^
--eval_h_t 0.05 ^
--activation Tanh ^
--use_sa ^
--sa_init_pde 1 ^
--sa_init_bc 1 ^
--sa_init_ini 1 ^
--sa_weight_mode square ^
--sa_optimizer adamw ^
--sa_lr 1e-4 ^
--optimizer adamw ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--num_epochs 10000 ^
--lbfgs_epochs 100 ^
--seeds 1 ^
--save_dir "runs/burgers2d/results_SA(burgers2d)"

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
 --pde burgers_2d ^
 --model_name PINN ^
 --layers 3 200 200 200 200 200 1 ^
 --use_causal ^
 --causal_epsilon_schedule 1 ^
 --causal_stage_epochs 10000 ^
 --causal_delta 0.99 ^
 --causal_normalization separate ^
 --no-causal_include_initial ^
 --lambda_pde 1.0 ^
 --lambda_bc 1.0 ^
 --lambda_ini 1.0 ^
 --h_x 0.0625 ^
 --h_y 0.0625 ^
 --h_t 0.0625 ^
 --eval_h_x 0.05 ^
 --eval_h_y 0.05 ^
 --eval_h_t 0.05 ^
 --lr 1e-3 ^
 --weight_decay 1e-6 ^
 --optimizer adamW ^
 --num_epochs 10000 ^
 --lbfgs_epochs 100 ^
 --seeds 1 2 3 4 5 ^
 --save_dir "runs/burgers2d/results_causal(burgers2d)"

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
--pde burgers_2d ^
--use_rba ^
--scheduler rba ^
--layers 3 200 200 200 200 200 1 ^
--h_x 0.0625 ^
--h_y 0.0625 ^
--h_t 0.0625 ^
--eval_h_x 0.05 ^
--eval_h_y 0.05 ^
--eval_h_t 0.05 ^
--lambda_pde 1.0 ^
--lambda_ini 1.0 ^
--lambda_bc 1.0 ^
--use_weight_norm ^
--activation Tanh ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--optimizer adamW ^
--num_epochs 10000 ^
--lbfgs_epochs 100 ^
--seeds 1 ^
--save_dir "runs/burgers2d/results_rba(burgers2d)"

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
--pde burgers_2d ^
--use_brdr ^
--scheduler brdr ^
--layers 3 200 200 200 200 200 1 ^
--h_x 0.0625 ^
--h_y 0.0625 ^
--h_t 0.0625 ^
--eval_h_x 0.05 ^
--eval_h_y 0.05 ^
--eval_h_t 0.05 ^
--lambda_pde 1.0 ^
--lambda_ini 1.0 ^
--lambda_bc 1.0 ^
--lr 1e-3 ^
--weight_decay 1e-6 ^
--optimizer adamW ^
--use_weight_norm ^
--activation Tanh ^
--num_epochs 10000 ^
--lbfgs_epochs 100 ^
--seeds 1 ^
--save_dir "runs/burgers2d/results_brdr(burgers2d)"

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
--seeds 1 ^
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
--seeds 1 ^
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
--seeds 1 ^
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
--seeds 1 ^
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
