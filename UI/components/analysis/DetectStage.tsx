'use client'

import { AlertTriangle, CheckCircle2 } from 'lucide-react'
import { categoryLabel } from '@/lib/constants'
import { DetectionResult } from '@/types'

interface DetectStageProps {
  result: DetectionResult | null
  loading: boolean
  error: string | null
}

export function DetectStage({ result, loading, error }: DetectStageProps) {
  if (loading) {
    return (
      <div className="rounded-lg bg-bg-2 border border-border-subtle px-4 py-8 text-center text-sm text-text-3">
        Analyzing…
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-lg bg-bg-2 border border-status-red border-opacity-30 px-4 py-4 text-sm text-status-red">
        {error}
      </div>
    )
  }

  if (!result) {
    return (
      <div className="rounded-lg bg-bg-2 border border-border-subtle border-dashed px-4 py-8 text-center text-sm text-text-4">
        Submit a passage above to see category results here.
      </div>
    )
  }

  const entries = Object.entries(result.categories)
  const flagged = entries.filter(([, v]) => v.flagged)
  const notFlagged = entries.filter(([, v]) => !v.flagged)

  return (
    <div className="rounded-lg bg-bg-2 border border-border-subtle overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        {flagged.length === 0 ? (
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-2xl text-sm font-medium border bg-status-green bg-opacity-10 border-status-green border-opacity-30 text-status-green">
            <CheckCircle2 size={14} />
            <span>No bias flagged across {entries.length} categories</span>
          </div>
        ) : (
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-2xl text-sm font-medium border bg-status-red bg-opacity-10 border-status-red border-opacity-30 text-status-red">
            <AlertTriangle size={14} />
            <span>
              {flagged.length} of {entries.length} categories flagged
            </span>
          </div>
        )}
      </div>

      {flagged.length > 0 && (
        <div className="flex flex-wrap gap-2 px-4 py-3 border-b border-border">
          {flagged.map(([category]) => (
            <span
              key={category}
              className="px-2.5 py-1 rounded-full text-xs font-medium bg-status-red bg-opacity-10 border border-status-red border-opacity-30 text-status-red"
            >
              {categoryLabel(category)}
            </span>
          ))}
        </div>
      )}

      {notFlagged.length > 0 && (
        <div className="px-4 py-3">
          <p className="text-[11px] uppercase tracking-wider text-text-4 mb-2">Not flagged</p>
          <div className="flex flex-wrap gap-1.5">
            {notFlagged.map(([category]) => (
              <span key={category} className="px-2 py-0.5 rounded-full text-xs text-text-4 bg-bg-3">
                {categoryLabel(category)}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
