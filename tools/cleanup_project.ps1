# Fixed cleanup plan for this Lab1 project. Default: preview only.
[CmdletBinding()]
param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$cleanupRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$expectedRoot = 'D:\人工智慧醫學影像應用\lab1\LAB1_315553010_楊敦傑'
if ($cleanupRoot -ne $expectedRoot) { throw 'Run this script only in the intended D drive project.' }
if ((Get-Item -LiteralPath $cleanupRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Project root is a reparse point.' }

$currentResults = Join-Path $cleanupRoot 'results\corrected_group_split_20261002'
$reportPath = Join-Path $cleanupRoot 'LAB1_315553010_楊敦傑.docx'
$proofPath = Join-Path $currentResults 'report_verification.json'
$proof = Get-Content -LiteralPath $proofPath -Raw | ConvertFrom-Json
if (-not ($proof.docx_zip_crc_ok -and $proof.all_selected_result_table_values_match -and
    $proof.all_embedded_figures_match_corrected_sources -and $proof.discussion_matches_corrected_data)) {
    throw 'Current Word content verification must pass before cleanup.'
}
if ((Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $proof.report_sha256) {
    throw 'Word has changed since verification. Verify it again and update the report proof before cleanup.'
}
$completion = Get-Content -LiteralPath (Join-Path $currentResults 'experiment_completed.json') -Raw | ConvertFrom-Json
if (-not ($completion.all_three_models_complete -and $completion.gpu_checkpoint_audit_passed -and $completion.pretrained_verification_passed)) {
    throw 'Formal experiment completion checks have not passed.'
}

# Only named obsolete artifacts are selected. No deletion of whole data/results roots.
$relativeTargets = @(
    'archive_from_c', 'report_backups', 'report_qa', '__pycache__',
    'data\cuda_cache', 'data\mplcache', 'data\temp',
    'data\source_verification\chest-xray-pneumonia-v2.zip',
    'results\resnet101',
    'audit_experiment.py', 'diagnose_split.py', 'plot_results.py',
    'report_discussion.py', 'edit_report_discussion.py', 'update_project_docs.py',
    'write_dataset_review.py', 'EXPERIMENT_REVIEW_20261002.md', 'MIGRATION_NOTES.md',
    'cleanup_project_sources.py',
    'results\accuracy_curve.png', 'results\accuracy_f1_curves.png',
    'results\audit_20261002.json', 'results\combined_epoch_metrics.csv',
    'results\comparison_accuracy.png', 'results\comparison_f1.png',
    'results\comparison_final_test_confusion.png', 'results\comparison_highest_test_confusion.png',
    'results\comparison_log_excerpt.png', 'results\comparison_summary.json',
    'results\comparison_validation.png', 'results\epoch_metrics.csv',
    'results\f1_curve.png', 'results\pretrained_verification.json',
    'results\resnet101_training.log', 'results\resnet18_best_validation.pt',
    'results\resnet18_final_test_confusion.png', 'results\resnet18_highest_test_accuracy_confusion.png',
    'results\resnet18_resnet50_training.log', 'results\resnet50_best_validation.pt',
    'results\resnet50_final_test_confusion.png', 'results\resnet50_highest_test_accuracy_confusion.png',
    'results\result_log_excerpt.png', 'results\run_metadata.json', 'results\summary.json'
)
$protectedPaths = @('.venv','data\chest_xray','data\torch_cache',
    'results\corrected_group_split_20261002','LAB1_315553010_楊敦傑.docx','.git') |
    ForEach-Object { [IO.Path]::GetFullPath((Join-Path $cleanupRoot $_)) }

function Assert-SafeTarget([string]$path) {
    $fullPath = [IO.Path]::GetFullPath($path).TrimEnd('\')
    if (-not $fullPath.StartsWith($cleanupRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside project: $fullPath"
    }
    foreach ($protected in $protectedPaths) {
        if ($fullPath -eq $protected -or $fullPath.StartsWith($protected + '\', [StringComparison]::OrdinalIgnoreCase) -or
            $protected.StartsWith($fullPath + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Protected path: $fullPath"
        }
    }
    # Check the target and its ancestors for junctions/symlinks.
    $ancestor = Get-Item -LiteralPath $fullPath -Force
    while ($ancestor -and $ancestor.FullName -ne $cleanupRoot) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $($ancestor.FullName)" }
        $ancestor = Get-Item -LiteralPath ([IO.Path]::GetDirectoryName($ancestor.FullName)) -Force
    }
    $targetItem = Get-Item -LiteralPath $fullPath -Force
    if ($targetItem.PSIsContainer) {
        foreach ($child in (Get-ChildItem -LiteralPath $fullPath -Force -Recurse)) {
            if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $($child.FullName)" }
        }
    }
    return $fullPath
}

$plan = @()
foreach ($relative in $relativeTargets) {
    $candidate = Join-Path $cleanupRoot $relative
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    $safePath = Assert-SafeTarget $candidate
    $item = Get-Item -LiteralPath $safePath -Force
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $safePath -File -Force -Recurse) } else { @($item) }
    $byteCount = ($files | Measure-Object Length -Sum).Sum
    if ($null -eq $byteCount) { $byteCount = 0 }
    $plan += [pscustomobject]@{RelativePath=$relative;AbsolutePath=$safePath;Directory=[bool]$item.PSIsContainer;Files=$files.Count;Bytes=[long]$byteCount}
}
$plan | Select-Object RelativePath,Files,@{Name='GiB';Expression={[math]::Round($_.Bytes/1GB,3)}} | Format-Table -AutoSize
$totalBytes = [long](($plan | Measure-Object Bytes -Sum).Sum)
Write-Output ("Planned removal: {0} targets, {1:N2} GiB." -f $plan.Count,($totalBytes/1GB))
if (-not $Execute) {
    Write-Output 'Preview only. To delete exactly these targets, run this script with -Execute.'
    return
}

$removed = @()
foreach ($entry in $plan) {
    # Revalidate immediately before each file or recursive directory deletion.
    $safePath = Assert-SafeTarget $entry.AbsolutePath
    if ($entry.Directory) {
        Remove-Item -LiteralPath $safePath -Recurse -Force
    } else {
        Remove-Item -LiteralPath $safePath -Force
    }
    $removed += $entry
    # Persist progress so an interrupted cleanup remains auditable.
    $cleanupRecord = [ordered]@{
        timestamp=(Get-Date).ToString('o'); project=$cleanupRoot
        removed=$removed; removed_bytes=[long](($removed | Measure-Object Bytes -Sum).Sum)
        completed=($removed.Count -eq $plan.Count)
        original_images_preserved=$true; formal_results_preserved=$true
        report_sha256=$proof.report_sha256
    }
    $cleanupRecord | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath (Join-Path $currentResults 'cleanup_summary.json') -Encoding UTF8
}
Write-Output ("Cleanup complete: {0} targets removed, {1:N2} GiB." -f $removed.Count,($totalBytes/1GB))

