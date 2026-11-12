param(
 [Parameter(Mandatory=$true)][ValidateSet('prepare','submit','status')][string]$Action,
 [Parameter(ValueFromRemainingArguments=$true)][string[]]$Options
)
& py -3 -X utf8 "$PSScriptRoot/submit_job.py" $Action @Options
exit $LASTEXITCODE
