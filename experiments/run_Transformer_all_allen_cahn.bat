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
--seeds 1 2 3 4 5 ^
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
--seeds 1 2 3 4 5 ^
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

pause
