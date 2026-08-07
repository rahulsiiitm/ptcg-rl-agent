param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]] $ReplayPaths
)

$ErrorActionPreference = "Stop"

$cardNames = @{
    6 = "Fighting Energy"
    673 = "Makuhita"
    674 = "Hariyama"
    675 = "Lunatone"
    676 = "Solrock"
    677 = "Riolu"
    678 = "Mega Lucario ex"
    1102 = "Dusk Ball"
    1123 = "Switch"
    1141 = "Premium Power Pro"
    1142 = "Fighting Gong"
    1152 = "Poke Pad"
    1159 = "Hero's Cape"
    1182 = "Boss's Orders"
    1192 = "Carmine"
    1227 = "Lillie's Determination"
    1252 = "Gravity Mountain"
}

function Get-CardName([int] $Id) {
    if ($cardNames.ContainsKey($Id)) { return $cardNames[$Id] }
    return "#$Id"
}

function Get-CardIds($Cards) {
    return @($Cards | Where-Object { $null -ne $_ -and $null -ne $_.id } | ForEach-Object { [int]$_.id })
}

function Get-EnergyGoal([int] $CardId) {
    if ($CardId -in @(673, 674)) { return 3 }
    if ($CardId -in @(677, 678)) { return 2 }
    if ($CardId -eq 676) { return 1 }
    return 0
}

function Get-SelectedOption($Visual) {
    if ($null -eq $Visual.selected -or $Visual.selected.Count -eq 0) { return $null }
    $index = [int]$Visual.selected[0]
    if ($index -lt 0 -or $index -ge $Visual.select.option.Count) { return $null }
    return $Visual.select.option[$index]
}

$optionNames = @{
    0 = "Number"; 1 = "Yes"; 2 = "No"; 3 = "Card"; 4 = "ToolCard"
    5 = "EnergyCard"; 6 = "Energy"; 7 = "Play"; 8 = "Attach"
    9 = "Evolve"; 10 = "Ability"; 11 = "Discard"; 12 = "Retreat"
    13 = "Attack"; 14 = "End"; 15 = "Skill"; 16 = "SpecialCondition"
}

$files = foreach ($path in $ReplayPaths) {
    Get-ChildItem -LiteralPath $path -ErrorAction SilentlyContinue
    if ($path.Contains("*")) { Get-ChildItem -Path $path -ErrorAction SilentlyContinue }
}
$files = @($files | Where-Object { $_.Extension -eq ".json" -and $_.Length -gt 10000 } |
    Sort-Object FullName -Unique)

$summary = @()
$allIssues = @()
$seenEpisodes = @{}

foreach ($file in $files) {
    $data = Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json
    if ($null -eq $data.steps -or $null -eq $data.info.TeamNames) { continue }
    if ($seenEpisodes.ContainsKey([string]$data.info.EpisodeId)) { continue }
    $seenEpisodes[[string]$data.info.EpisodeId] = $true

    $ourIndex = 0
    for ($i = 0; $i -lt $data.info.TeamNames.Count; $i++) {
        if ([string]$data.info.TeamNames[$i] -match "(?i)rahul") { $ourIndex = $i; break }
    }
    $opIndex = 1 - $ourIndex
    $reward = [int]$data.rewards[$ourIndex]
    $result = if ($reward -gt 0) { "WIN" } elseif ($reward -lt 0) { "LOSS" } else { "DRAW" }

    # Only top-level step actions were executed in the real match. The replay's
    # `visualize` arrays also contain internal search branches, so treating their
    # `selected` fields as real actions produces impossible false positives.
    $ourActions = @()
    # Kaggle records the action on step N and the observation it answered on
    # step N-1. Decoding an action against step N's new options assigns the
    # same numeric index to the wrong move.
    for ($stepIndex = 1; $stepIndex -lt $data.steps.Count; $stepIndex++) {
        $step = $data.steps[$stepIndex]
        $previousStep = $data.steps[$stepIndex - 1]
        if ($step.Count -le $ourIndex -or $previousStep.Count -le $ourIndex) { continue }
        $agentStep = $step[$ourIndex]
        $observation = $previousStep[$ourIndex].observation
        $action = @($agentStep.action)
        if ($null -eq $observation.current -or $null -eq $observation.select -or $action.Count -eq 0) {
            continue
        }
        $selectedIndex = [int]$action[0]
        if ($selectedIndex -lt 0 -or $selectedIndex -ge $observation.select.option.Count) { continue }
        $ourActions += [pscustomobject]@{
            Current = $observation.current
            Select = $observation.select
            SelectedIndex = $selectedIndex
            Chosen = $observation.select.option[$selectedIndex]
        }
    }

    $main = @($ourActions | Where-Object { [int]$_.Select.context -eq 0 })
    $issues = @()

    foreach ($turnGroup in ($main | Group-Object { [int]$_.Current.turn })) {
        $turn = [int]$turnGroup.Name
        $decisions = @($turnGroup.Group)
        $chosenTypes = @($decisions | ForEach-Object { [int]$_.Chosen.type })
        $allOptionTypes = @($decisions | ForEach-Object { $_.Select.option } | ForEach-Object { [int]$_.type })
        $lastDecision = $decisions | Select-Object -Last 1
        $me = $lastDecision.Current.players[$ourIndex]
        $active = @($me.active | Where-Object { $null -ne $_ }) | Select-Object -First 1
        $activeId = if ($null -ne $active) { [int]$active.id } else { -1 }
        $activeEnergy = if ($null -ne $active) { @($active.energyCards).Count } else { 0 }

        # Judge the entire real turn, not an intermediate decision. Some cards
        # produce another Main selection after an End/Attack-like option.
        if ($allOptionTypes -contains 13 -and $chosenTypes -notcontains 13) {
            $issues += [pscustomobject]@{
                Episode = $data.info.EpisodeId; Result = $result; Turn = $turn
                Type = "ATTACK_SKIPPED"; Detail = "Attack was legal during the turn but never chosen; final active $(Get-CardName $activeId) had $activeEnergy energy."
            }
        }

        $attachOpportunity = $false
        foreach ($decision in $decisions) {
            $handIds = Get-CardIds $decision.Current.players[$ourIndex].hand
            $optionTypes = @($decision.Select.option | ForEach-Object { [int]$_.type })
            if (-not [bool]$decision.Current.energyAttached -and $handIds -contains 6 -and $optionTypes -contains 8) {
                foreach ($option in $decision.Select.option) {
                    if ([int]$option.type -ne 8) { continue }
                    $target = if ([int]$option.inPlayArea -eq 4) {
                        @($decision.Current.players[$ourIndex].active)[[int]$option.inPlayIndex]
                    } elseif ([int]$option.inPlayArea -eq 5) {
                        @($decision.Current.players[$ourIndex].bench)[[int]$option.inPlayIndex]
                    } else { $null }
                    if ($null -ne $target -and @($target.energyCards).Count -lt (Get-EnergyGoal ([int]$target.id))) {
                        $attachOpportunity = $true
                    }
                }
            }
        }
        if ($attachOpportunity -and $chosenTypes -notcontains 8) {
            $issues += [pscustomobject]@{
                Episode = $data.info.EpisodeId; Result = $result; Turn = $turn
                Type = "ATTACH_SKIPPED"; Detail = "Fighting Energy could be attached during the turn but no Attach was chosen."
            }
        }

        $bossOpportunity = $null
        $bossChosen = $false
        foreach ($decision in $decisions) {
            $player = $decision.Current.players[$ourIndex]
            $opponent = $decision.Current.players[$opIndex]
            if ([int]$decision.Chosen.type -eq 7) {
                $chosenCard = $player.hand[[int]$decision.Chosen.index]
                if ($null -ne $chosenCard -and [int]$chosenCard.id -eq 1182) { $bossChosen = $true }
            }
            if (-not [bool]$decision.Current.supporterPlayed) {
                foreach ($option in $decision.Select.option) {
                    if ([int]$option.type -ne 7) { continue }
                    $card = $player.hand[[int]$option.index]
                    if ($null -eq $card -or [int]$card.id -ne 1182) { continue }
                    $opponentActive = @($opponent.active | Where-Object { $null -ne $_ }) | Select-Object -First 1
                    # A damaged bench target is not a Boss opportunity when
                    # the Active is already in Aura Jab KO range.
                    if ($null -ne $opponentActive -and [int]$opponentActive.hp -le 130) { continue }
                    $bossOpportunity = @($opponent.bench | Where-Object {
                        $null -ne $_ -and [int]$_.maxHp -gt 0 -and ([double]$_.hp / [double]$_.maxHp) -le 0.40
                    } | Sort-Object hp | Select-Object -First 1)
                }
            }
        }
        if ($null -ne $bossOpportunity -and $bossOpportunity.Count -gt 0 -and -not $bossChosen) {
            $target = $bossOpportunity[0]
            $issues += [pscustomobject]@{
                Episode = $data.info.EpisodeId; Result = $result; Turn = $turn
                Type = "BOSS_MISSED"; Detail = "Boss was legal with bench target $(Get-CardName ([int]$target.id)) at $($target.hp)/$($target.maxHp), but was not chosen."
            }
        }
    }

    # Resolution contexts are where v21's numeric enum assumptions caused its largest regression.
    $resolution = @($ourActions | Where-Object { [int]$_.Select.context -ne 0 })
    $discardChoices = @($resolution | Where-Object { [int]$_.Chosen.type -eq 11 }).Count
    $yesChoices = @($resolution | Where-Object { [int]$_.Chosen.type -eq 1 }).Count
    $noChoices = @($resolution | Where-Object { [int]$_.Chosen.type -eq 2 }).Count

    $last = $ourActions | Select-Object -Last 1
    $opponentVisibleIds = @()
    if ($null -ne $last) {
        $op = $last.Current.players[$opIndex]
        $opponentVisibleIds = Get-CardIds (@($op.active) + @($op.bench) + @($op.discard))
    }
    $topOpponent = @($opponentVisibleIds | Group-Object | Sort-Object Count -Descending |
        Select-Object -First 5 | ForEach-Object { "$(Get-CardName ([int]$_.Name))x$($_.Count)" }) -join ", "

    $summary += [pscustomobject]@{
        Episode = $data.info.EpisodeId
        Result = $result
        Opponent = $data.info.TeamNames[$opIndex]
        Decisions = $ourActions.Count
        Main = $main.Count
        Discard = $discardChoices
        Yes = $yesChoices
        No = $noChoices
        Issues = $issues.Count
        OpponentCards = $topOpponent
    }
    $allIssues += $issues
}

$summary | Sort-Object Episode | Format-Table -AutoSize
""
"Results: $(@($summary | Where-Object Result -eq 'WIN').Count) wins, $(@($summary | Where-Object Result -eq 'LOSS').Count) losses, $(@($summary | Where-Object Result -eq 'DRAW').Count) draws"
""
"Issue totals:"
$allIssues | Group-Object Type | Sort-Object Count -Descending | Format-Table Count, Name -AutoSize
""
"Issues in losses:"
$allIssues | Where-Object Result -eq "LOSS" | Sort-Object Episode, Turn | Format-Table Episode, Turn, Type, Detail -AutoSize -Wrap
