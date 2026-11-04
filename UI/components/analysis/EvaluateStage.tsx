'use client'

import { categoryLabel, categoryShortLabel } from '@/lib/constants'
import { RadarChart, RadarChartDatum, RadarChartLegend, PerformanceTier } from '@/components/ui'
import { EvalMetrics } from '@/types'

interface EvaluateStageProps {
  metrics: EvalMetrics | null
  loading: boolean
}

// Below this many test examples, a category's score is too noisy to trust --
// a single right/wrong guess swings it by 10-50%. Shown as its own "Limited
// data" tier rather than folded into "weak," since a low score from too
// little data is a data problem, not a sign the model failed at that category.
const LOW_SUPPORT_THRESHOLD = 10

function tierFor(f1: number, support: number): PerformanceTier {
  if (support < LOW_SUPPORT_THRESHOLD) return 'limited-data'
  if (f1 >= 0.6) return 'strong'
  if (f1 >= 0.3) return 'mixed'
  return 'weak'
}

export function EvaluateStage({ metrics, loading }: EvaluateStageProps) {
  if (loading) {
    return (
      <div className="rounded-lg bg-bg-2 border border-border-subtle px-4 py-8 text-center text-sm text-text-3">
        Loading evaluation results…
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="rounded-lg bg-bg-2 border border-border-subtle border-dashed px-4 py-8 text-center text-sm text-text-4">
        No evaluation snapshot yet — run{' '}
        <code className="text-text-3">python src/training/evaluate_llm_lora.py</code> to generate one.
      </div>
    )
  }

  const { binary_presence, dataset } = metrics
  const wholePicturePct = Math.round(binary_presence.overall.all_categories_exact_match * 100)
  const allAvgQualityPct = Math.round(binary_presence.overall.macro_f1_across_categories * 100)

  const chartData: RadarChartDatum[] = binary_presence.per_category.map((c) => ({
    key: c.category,
    shortLabel: categoryShortLabel(c.category),
    fullLabel: categoryLabel(c.category),
    value: c.f1,
    support: c.support,
    tier: tierFor(c.f1, c.support),
  }))

  const lowSupportCategories = binary_presence.per_category.filter((c) => c.support < LOW_SUPPORT_THRESHOLD)
  const supportedCategories = binary_presence.per_category.filter((c) => c.support >= LOW_SUPPORT_THRESHOLD)
  const supportedAvgQualityPct =
    supportedCategories.length > 0
      ? Math.round((supportedCategories.reduce((sum, c) => sum + c.f1, 0) / supportedCategories.length) * 100)
      : null

  const sortedByF1 = [...binary_presence.per_category].sort((a, b) => b.f1 - a.f1)

  return (
    <div className="rounded-lg bg-bg-2 border border-border-subtle overflow-hidden">
      {/* Plain-language summary */}
      <div className="px-4 py-4 border-b border-border">
        <p className="text-sm text-text-1 leading-relaxed">
          On text it had never seen before, the model got <strong>every one</strong> of the 14 categories
          right at the same time <strong className="text-accent-text">{wholePicturePct}%</strong> of the
          time.
        </p>
        <p className="mt-2 text-sm text-text-1 leading-relaxed">
          Averaged across <strong>all 14 categories</strong>, detection quality scores{' '}
          <strong className="text-accent-text">{allAvgQualityPct}%</strong>.
          {supportedAvgQualityPct !== null && (
            <>
              {' '}
              Looking only at the <strong>{supportedCategories.length} categories with enough test data to
              trust the score</strong> ({LOW_SUPPORT_THRESHOLD}+ examples), that rises to{' '}
              <strong className="text-accent-text">{supportedAvgQualityPct}%</strong>.
            </>
          )}
        </p>
        <p className="mt-1.5 text-xs text-text-4">
          Tested on {dataset.test} examples the model wasn&apos;t trained on (trained on {dataset.train},
          tuned on {dataset.val}).
        </p>
      </div>

      {/* Radar chart */}
      <div className="px-4 pt-5 pb-3">
        <RadarChart data={chartData} />
      </div>
      <div className="px-4 pb-4">
        <RadarChartLegend />
      </div>

      {lowSupportCategories.length > 0 && (
        <div className="px-4 py-3 border-t border-border bg-bg-3 bg-opacity-40">
          <p className="text-xs text-text-3 leading-relaxed">
            <strong className="text-text-2">
              {lowSupportCategories.length} categories are marked &ldquo;Limited data&rdquo;
            </strong>{' '}
            ({lowSupportCategories.map((c) => categoryLabel(c.category)).join(', ')}) — fewer than{' '}
            {LOW_SUPPORT_THRESHOLD} real-world examples of that bias type showed up in the test set. A low
            score there reflects how rare that category is in the data, not necessarily a weak model.
          </p>
        </div>
      )}

      {/* Detailed numbers, collapsed by default */}
      <details className="border-t border-border">
        <summary className="px-4 py-3 text-xs font-medium text-text-3 cursor-pointer select-none hover:text-text-2">
          Show detailed numbers per category
        </summary>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-text-4 uppercase tracking-wider">
                <th className="text-left font-medium px-4 py-2">Category</th>
                <th className="text-right font-medium px-4 py-2">Precision*</th>
                <th className="text-right font-medium px-4 py-2">Recall**</th>
                <th className="text-right font-medium px-4 py-2">F1***</th>
                <th className="text-right font-medium px-4 py-2">Test examples</th>
              </tr>
            </thead>
            <tbody>
              {sortedByF1.map((c) => (
                <tr key={c.category} className="border-t border-border">
                  <td className="px-4 py-2 text-text-2">{categoryLabel(c.category)}</td>
                  <td className="px-4 py-2 text-right text-text-3 tabular-nums">{c.precision.toFixed(2)}</td>
                  <td className="px-4 py-2 text-right text-text-3 tabular-nums">{c.recall.toFixed(2)}</td>
                  <td className="px-4 py-2 text-right text-text-3 tabular-nums">{c.f1.toFixed(2)}</td>
                  <td className="px-4 py-2 text-right text-text-4 tabular-nums">{c.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <dl className="px-4 py-3 border-t border-border space-y-1 text-[11px] text-text-4">
          <div>
            <dt className="inline font-medium text-text-3">* Precision —</dt>{' '}
            <dd className="inline">Of the times it said &ldquo;biased,&rdquo; how often was it actually right?</dd>
          </div>
          <div>
            <dt className="inline font-medium text-text-3">** Recall —</dt>{' '}
            <dd className="inline">Of all the truly biased examples, how many did it catch?</dd>
          </div>
          <div>
            <dt className="inline font-medium text-text-3">*** F1 —</dt>{' '}
            <dd className="inline">
              A single balanced score combining precision and recall (0 = worst, 1 = best) — the number the
              table is sorted by.
            </dd>
          </div>
        </dl>
      </details>

      <p className="px-4 py-2 text-[11px] text-text-4 border-t border-border">
        Generated {new Date(metrics.generated_at).toLocaleString()} · {metrics.model} · scored on whether
        each category was correctly flagged as biased or not (precision/recall/F1 on presence, not exact
        severity level).
      </p>
    </div>
  )
}
