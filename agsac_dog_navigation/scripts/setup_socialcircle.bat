@echo off
chcp 65001 >nul
REM 设置UTF-8编码，避免中文乱码

echo ========================================
echo SocialCircle Integration Script
echo ========================================

REM 设置路径
set PROJECT_ROOT=%~dp0..
set EXTERNAL_DIR=%PROJECT_ROOT%\external
set TEMP_DIR=%PROJECT_ROOT%\..\SocialCircle_temp

echo.
echo Step 1: Checking for existing temp directory...
if exist "%TEMP_DIR%" (
    echo [WARNING] Temp directory exists, removing...
    rmdir /s /q "%TEMP_DIR%"
)

echo.
echo Step 2: Clone SocialCircle repository...
cd %PROJECT_ROOT%\..

REM 先尝试clone主仓库
echo Cloning SocialCircle repository...
git clone https://github.com/cocoon2wong/SocialCircle.git SocialCircle_temp

if errorlevel 1 (
    echo [ERROR] Clone failed. Please check network connection.
    pause
    exit /b 1
)

REM 进入仓库查看可用分支
cd SocialCircle_temp
echo.
echo Available branches:
git branch -r

REM 尝试切换到PyTorch分支（尝试多个可能的名称）
echo.
echo Attempting to switch to PyTorch branch...

git checkout "TorchVersion(beta)" 2>nul
if not errorlevel 1 (
    echo [OK] Switched to TorchVersion(beta) branch
    goto :branch_found
)

git checkout TorchVersion 2>nul
if not errorlevel 1 (
    echo [OK] Switched to TorchVersion branch
    goto :branch_found
)

git checkout pytorch 2>nul
if not errorlevel 1 (
    echo [OK] Switched to pytorch branch
    goto :branch_found
)

git checkout torch 2>nul
if not errorlevel 1 (
    echo [OK] Switched to torch branch
    goto :branch_found
)

REM 如果都失败，提示用户
echo.
echo [WARNING] Could not find PyTorch branch automatically.
echo Please check the available branches above and manually:
echo 1. cd %TEMP_DIR%
echo 2. git checkout [pytorch-branch-name]
echo 3. Re-run this script
echo.
echo Or continue with main branch (TensorFlow version)
echo Press any key to continue with main branch, or Ctrl+C to exit...
pause >nul

:branch_found
cd %PROJECT_ROOT%\..

echo.
echo Step 3: Creating external directory...
if not exist "%EXTERNAL_DIR%" mkdir "%EXTERNAL_DIR%"
if not exist "%EXTERNAL_DIR%\SocialCircle_original" mkdir "%EXTERNAL_DIR%\SocialCircle_original"

echo.
echo Step 4: Copying core code...
xcopy /E /I /Y "%TEMP_DIR%\socialCircle" "%EXTERNAL_DIR%\SocialCircle_original\socialCircle"
xcopy /E /I /Y "%TEMP_DIR%\qpid" "%EXTERNAL_DIR%\SocialCircle_original\qpid"

REM Copy dependency files
copy /Y "%TEMP_DIR%\requirements.txt" "%EXTERNAL_DIR%\SocialCircle_original\requirements.txt" 2>nul
copy /Y "%TEMP_DIR%\README.md" "%EXTERNAL_DIR%\SocialCircle_original\README.md" 2>nul

echo.
echo Step 5: Cleaning up temp directory...
cd %PROJECT_ROOT%
rmdir /s /q "%TEMP_DIR%"

echo.
echo ========================================
echo [SUCCESS] SocialCircle code copied to:
echo %EXTERNAL_DIR%\SocialCircle_original
echo ========================================
echo.
echo Next steps:
echo 1. Download pretrained weights to pretrained/social_circle/
echo 2. Run adapter test: python -m agsac.models.encoders.social_circle_pretrained
echo.
pause

