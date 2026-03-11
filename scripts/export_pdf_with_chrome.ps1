param(
    [Parameter(Mandatory = $true)]
    [string]$HtmlPath,
    [Parameter(Mandatory = $true)]
    [string]$PdfPath
)

$ErrorActionPreference = "Stop"

$chromeCandidates = @(
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
)

$chrome = $null
foreach ($candidate in $chromeCandidates) {
    if (Test-Path $candidate) {
        $chrome = $candidate
        break
    }
}

if (-not $chrome) {
    throw "Chrome executable not found in standard locations."
}

$resolvedHtml = (Resolve-Path $HtmlPath).Path
$resolvedPdfDir = Split-Path -Path $PdfPath -Parent
if ($resolvedPdfDir -and -not (Test-Path $resolvedPdfDir)) {
    New-Item -ItemType Directory -Path $resolvedPdfDir -Force | Out-Null
}
$resolvedPdf = [System.IO.Path]::GetFullPath($PdfPath)
$htmlUri = [System.Uri]::new($resolvedHtml).AbsoluteUri
$userDataDir = [System.IO.Path]::GetFullPath(".chrome_tmp_pdf")
if (-not (Test-Path $userDataDir)) {
    New-Item -ItemType Directory -Path $userDataDir -Force | Out-Null
}

if (Test-Path $resolvedPdf) {
    Remove-Item $resolvedPdf -Force
}

& $chrome --headless --no-sandbox --disable-gpu --disable-crash-reporter "--user-data-dir=$userDataDir" "--print-to-pdf=$resolvedPdf" $htmlUri

for ($i = 0; $i -lt 20; $i++) {
    if (Test-Path $resolvedPdf) {
        break
    }
    Start-Sleep -Milliseconds 250
}

if (-not (Test-Path $resolvedPdf)) {
    throw "PDF export failed: $resolvedPdf"
}

Get-Item $resolvedPdf | Select-Object FullName, Length, LastWriteTime
