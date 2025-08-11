Write-Host "==> Starting local Temporal pipeline test..."

# 1. Go into your project folder
Set-Location -Path "temporal_5"

# 2. Install Python requirements
Write-Host "==> Installing Python requirements..."
pip install -r requirements.txt

# 3. Start Temporal services using Docker Compose
Write-Host "==> Starting Temporal stack..."
docker-compose -f docker-compose.temporal.yaml up --build -d

Write-Host "==> Waiting for Temporal to be ready..."
Start-Sleep -Seconds 25

# 4. Build Docker image for processing
Write-Host "==> Building processing Docker image..."
docker build -t s2-product-pipeline:latest .

# 5. Run a test processing step
Write-Host "==> Running processing container (download step)..."
docker run -v "${PWD}:/app" s2-product-pipeline:latest `
  python run_one_product.py `
  --step download `
  --config config/config.yaml `
  --product-id S2A_MSIL2A_20220129T132231_N0510_R038_T23LLD_20240508T030029.SAFE

# 6. Start the Temporal worker in background
Write-Host "==> Starting Temporal worker in background..."
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "worker.py"

# 7. Trigger workflow
Write-Host "==> Triggering workflow..."
Start-Sleep -Seconds 5
python start_workflow.py

Write-Host "==> ✅ All steps completed!"
