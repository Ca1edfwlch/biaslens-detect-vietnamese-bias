'use client'

import { Sparkles, Wand2 } from 'lucide-react'
import { Button } from '@/components/ui'
import { DetectionResult, MitigationResult } from '@/types'

interface MitigateStageProps {
  detection: DetectionResult | null
  result: MitigationResult | null
  loading: boolean
  error: string | null
  onMitigate: () => void
}

export function MitigateStage({ detection, result, loading, error, onMitigate }: MitigateStageProps) {
  if (!detection) {
    return (
      <div className="rounded-lg bg-bg-2 border border-border-subtle border-dashed px-4 py-8 text-center text-sm text-text-4">
        Run detection above first.
      </div>
    )
  }

  const hasFlagged = Object.values(detection.categories).some((c) => c.flagged)

  if (!hasFlagged && !result) {
    return (
      <div className="rounded-lg bg-bg-2 border border-border-subtle border-dashed px-4 py-8 text-center text-sm text-text-4">
        No bias flagged — nothing to rewrite.
      </div>
    )
  }

  return (
    <div className="rounded-lg bg-bg-2 border border-border-subtle overflow-hidden">
      {result && (
        <div className="divide-y divide-border">
          <div className="px-4 py-3">
            <p className="text-[11px] uppercase tracking-wider text-text-4 mb-1.5">Original</p>
            <p className="text-sm text-text-3 leading-relaxed">{result.original_text}</p>
          </div>
          <div className="px-4 py-3 bg-accent bg-opacity-5">
            <p className="text-[11px] uppercase tracking-wider text-accent-text mb-1.5">Rewritten</p>
            <p className="text-sm text-text-1 leading-relaxed">{result.safer_text}</p>
          </div>
        </div>
      )}

      {error && (
        <div className="px-4 py-3 text-sm text-status-red border-t border-border first:border-t-0">
          {error}
        </div>
      )}

      <div className="flex items-center justify-between px-4 py-3 border-t border-border">
        <span className="text-xs text-text-4">
          {result ? `Rewritten with ${result.model}` : 'AI rewrite of the flagged passage'}
        </span>
        <Button
          icon={result ? <Wand2 size={15} /> : <Sparkles size={15} />}
          variant={result ? 'secondary' : 'primary'}
          size="sm"
          loading={loading}
          onClick={onMitigate}
        >
          {result ? 'Rewrite again' : 'Rewrite with AI'}
        </Button>
      </div>
    </div>
  )
}
