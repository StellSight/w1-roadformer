$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "没有找到 docker。请先安装并启动 Docker Desktop，再执行本脚本。"
}
if (-not (Test-Path ".\pretrained.pth")) {
    Write-Error "缺少 pretrained.pth。训练结束后把 best.pt 复制到项目根目录，文件名改为 pretrained.pth。"
}

docker build -t w1-roadformer:latest .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
docker save w1-roadformer:latest -o w1-roadformer.tar
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "已生成 $(Join-Path $PSScriptRoot 'w1-roadformer.tar')"
