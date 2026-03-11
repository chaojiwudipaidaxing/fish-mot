param(
    [Parameter(Mandatory = $true)] [string] $BaseDocx,
    [Parameter(Mandatory = $true)] [string] $ReviewDocx,
    [Parameter(Mandatory = $true)] [string] $GraphicalAbstractDocx,
    [Parameter(Mandatory = $true)] [string] $SinglePdf,
    [Parameter(Mandatory = $true)] [string] $ReviewPdf,
    [Parameter(Mandatory = $true)] [string] $TwoColumnDocx,
    [Parameter(Mandatory = $true)] [string] $TwoColumnPdf,
    [Parameter(Mandatory = $true)] [string] $GraphicalAbstractPdf
)

$ErrorActionPreference = "Stop"

$wdFormatPDF = 17
$wdSectionBreakContinuous = 3
$wdCollapseStart = 1
$wdCollapseEnd = 0

function Ensure-Parent {
    param([string] $PathValue)
    $parent = Split-Path -Parent $PathValue
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
}

function Open-WordDocument {
    param($WordApp, [string] $PathValue)
    return $WordApp.Documents.Open($PathValue)
}

function Save-WordDocument {
    param($Doc, [string] $PathValue)
    Ensure-Parent -PathValue $PathValue
    $Doc.SaveAs([ref] $PathValue)
}

function Export-WordPdf {
    param($Doc, [string] $PdfPath)
    Ensure-Parent -PathValue $PdfPath
    try {
        $Doc.Fields.Update() | Out-Null
    } catch {
    }
    $Doc.ExportAsFixedFormat($PdfPath, $wdFormatPDF)
}

function Find-TextRange {
    param($Doc, [string] $Text)
    $range = $Doc.Content
    $find = $range.Find
    $find.ClearFormatting() | Out-Null
    $find.Text = $Text
    $find.Forward = $true
    $find.Wrap = 0
    if ($find.Execute()) {
        return $range
    }
    return $null
}

function Insert-ContinuousSectionBeforeText {
    param($Doc, [string] $Text)
    $range = Find-TextRange -Doc $Doc -Text $Text
    if ($null -eq $range) {
        return $false
    }
    $range.Collapse($wdCollapseStart)
    $range.InsertBreak($wdSectionBreakContinuous)
    return $true
}

function Get-ParagraphAtPosition {
    param($Doc, [int] $Pos)
    $r = $Doc.Range($Pos, $Pos)
    if ($r.Paragraphs.Count -gt 0) {
        return $r.Paragraphs.Item(1)
    }
    return $null
}

function Paragraph-StyleName {
    param($Paragraph)
    if ($null -eq $Paragraph) {
        return ""
    }
    try {
        return [string] $Paragraph.Range.Style.NameLocal
    } catch {
        return ""
    }
}

function Wrap-RangeInContinuousSection {
    param($Doc, [int] $Start, [int] $End)
    if ($Start -lt 1 -or $End -lt $Start) {
        return
    }
    $endRange = $Doc.Range($End, $End)
    $endRange.InsertBreak($wdSectionBreakContinuous)
    $startRange = $Doc.Range($Start, $Start)
    $startRange.InsertBreak($wdSectionBreakContinuous)
}

function Build-TableBlocks {
    param($Doc)
    $blocks = @()
    for ($i = $Doc.Tables.Count; $i -ge 1; $i--) {
        $table = $Doc.Tables.Item($i)
        $start = [int] $table.Range.Start
        $end = [int] $table.Range.End

        $beforePara = Get-ParagraphAtPosition -Doc $Doc -Pos ([Math]::Max(1, $start - 1))
        if (Paragraph-StyleName -Paragraph $beforePara -eq "Table Caption") {
            $start = [int] $beforePara.Range.Start
        }

        $blocks += [pscustomobject]@{
            Start = $start
            End = $end
            Kind = "Table"
        }
    }
    return $blocks
}

function Build-FigureBlocks {
    param($Doc)
    $blocks = @()
    $captions = @(
        "Core method family comparison (mean±std).",
        "Strong baseline comparison (ByteTrack/OC-SORT/BoT-SORT).",
        "Combined comparison of proposed variants and strong baselines.",
        "Controlled degradation curves (Base, +gating, ByteTrack).",
        "Sensitivity of +gating to threshold values (1000/2000/4000) at seed=0.",
        "Stratified metrics over occlusion, density, turning, and low-confidence buckets; quantile and fallback notes are reported in stratified_metrics_val.csv.",
        "Count stability comparison.",
        "Runtime profiling (FPS, memory, normalized CPU)."
    )

    foreach ($caption in $captions) {
        $range = Find-TextRange -Doc $Doc -Text $caption
        if ($null -eq $range) {
            continue
        }

        $captionPara = $range.Paragraphs.Item(1)
        $start = [int] $captionPara.Range.Start
        $prevPara = $captionPara.Previous()
        if ($null -ne $prevPara -and (Paragraph-StyleName -Paragraph $prevPara) -eq "Captioned Figure") {
            $start = [int] $prevPara.Range.Start
        }

        $blocks += [pscustomobject]@{
            Start = $start
            End = [int] $captionPara.Range.End
            Kind = "Figure"
        }
    }

    return $blocks
}

function Set-SectionColumns {
    param($Section, [int] $Count)
    $Section.PageSetup.TextColumns.SetCount($Count)
    if ($Count -gt 1) {
        $Section.PageSetup.TextColumns.Spacing = 18
        $Section.PageSetup.TextColumns.EvenlySpaced = $true
        $Section.PageSetup.TextColumns.LineBetween = $false
    }
}

function Apply-TwoColumnLayout {
    param($Doc)

    [void] (Insert-ContinuousSectionBeforeText -Doc $Doc -Text "1. Introduction")

    $blocks = @()
    $blocks += Build-TableBlocks -Doc $Doc
    $blocks += Build-FigureBlocks -Doc $Doc
    $blocks = $blocks | Sort-Object Start -Descending

    foreach ($block in $blocks) {
        Wrap-RangeInContinuousSection -Doc $Doc -Start $block.Start -End $block.End
    }

    for ($i = 1; $i -le $Doc.Sections.Count; $i++) {
        $section = $Doc.Sections.Item($i)
        if ($i -eq 1) {
            Set-SectionColumns -Section $section -Count 1
        } else {
            Set-SectionColumns -Section $section -Count 2
        }
        $section.PageSetup.TopMargin = 50
        $section.PageSetup.BottomMargin = 55
        $section.PageSetup.LeftMargin = 42
        $section.PageSetup.RightMargin = 42
        $section.PageSetup.HeaderDistance = 22
        $section.PageSetup.FooterDistance = 22
    }

    for ($i = 1; $i -le $Doc.Tables.Count; $i++) {
        $tableSection = $Doc.Tables.Item($i).Range.Sections.Item(1)
        Set-SectionColumns -Section $tableSection -Count 1
        try {
            $Doc.Tables.Item($i).Rows.AllowBreakAcrossPages = 0
        } catch {
        }
        try {
            $Doc.Tables.Item($i).AutoFitBehavior(2) | Out-Null
        } catch {
        }
    }

    $figureCaptions = @(
        "Core method family comparison (mean±std).",
        "Strong baseline comparison (ByteTrack/OC-SORT/BoT-SORT).",
        "Combined comparison of proposed variants and strong baselines.",
        "Controlled degradation curves (Base, +gating, ByteTrack).",
        "Sensitivity of +gating to threshold values (1000/2000/4000) at seed=0.",
        "Stratified metrics over occlusion, density, turning, and low-confidence buckets; quantile and fallback notes are reported in stratified_metrics_val.csv.",
        "Count stability comparison.",
        "Runtime profiling (FPS, memory, normalized CPU)."
    )
    foreach ($caption in $figureCaptions) {
        $range = Find-TextRange -Doc $Doc -Text $caption
        if ($null -ne $range) {
            $section = $range.Sections.Item(1)
            Set-SectionColumns -Section $section -Count 1
        }
    }
}

function New-WordApp {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    return $wordApp
}

function Close-WordApp {
    param($WordApp)
    if ($null -eq $WordApp) {
        return
    }
    try { $WordApp.Quit() } catch {}
    try { [System.Runtime.Interopservices.Marshal]::ReleaseComObject($WordApp) | Out-Null } catch {}
    [gc]::Collect()
    [gc]::WaitForPendingFinalizers()
}

Write-Host "[1/4] Export single-column PDF"
$word = $null
$doc = $null
try {
    $word = New-WordApp
    $doc = Open-WordDocument -WordApp $word -PathValue $BaseDocx
    Export-WordPdf -Doc $doc -PdfPath $SinglePdf
}
finally {
    if ($null -ne $doc) { try { $doc.Close([ref] $false) } catch {} }
    Close-WordApp -WordApp $word
}

Write-Host "[2/4] Export review PDF"
$word = $null
$doc = $null
try {
    $word = New-WordApp
    $doc = Open-WordDocument -WordApp $word -PathValue $ReviewDocx
    Export-WordPdf -Doc $doc -PdfPath $ReviewPdf
}
finally {
    if ($null -ne $doc) { try { $doc.Close([ref] $false) } catch {} }
    Close-WordApp -WordApp $word
}

Write-Host "[3/4] Build and export two-column variant"
$word = $null
$doc = $null
try {
    Copy-Item -LiteralPath $BaseDocx -Destination $TwoColumnDocx -Force
    $word = New-WordApp
    $doc = Open-WordDocument -WordApp $word -PathValue $TwoColumnDocx
    Apply-TwoColumnLayout -Doc $doc
    Save-WordDocument -Doc $doc -PathValue $TwoColumnDocx
    Export-WordPdf -Doc $doc -PdfPath $TwoColumnPdf
}
finally {
    if ($null -ne $doc) { try { $doc.Close([ref] $false) } catch {} }
    Close-WordApp -WordApp $word
}

Write-Host "[4/4] Export graphical abstract PDF"
$word = $null
$doc = $null
try {
    $word = New-WordApp
    $doc = Open-WordDocument -WordApp $word -PathValue $GraphicalAbstractDocx
    Export-WordPdf -Doc $doc -PdfPath $GraphicalAbstractPdf
}
finally {
    if ($null -ne $doc) { try { $doc.Close([ref] $false) } catch {} }
    Close-WordApp -WordApp $word
}
