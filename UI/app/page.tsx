'use client'

import { useEffect, useState } from 'react'
import { Scale } from 'lucide-react'
import { detectBias, getEvalMetrics, mitigateBias } from '@/lib/api'
import { categoryLabel } from '@/lib/constants'
import { DetectionResult, EvalMetrics, MitigationResult } from '@/types'
import { InputStage } from '@/components/analysis/InputStage'
import { DetectStage } from '@/components/analysis/DetectStage'
import { MitigateStage } from '@/components/analysis/MitigateStage'
import { EvaluateStage } from '@/components/analysis/EvaluateStage'

const STAGES = [
  { n: 1, title: 'Input' },
  { n: 2, title: 'Detect' },
  { n: 3, title: 'Mitigate' },
  { n: 4, title: 'Evaluate' },
] as const

export default function Home() {
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [detecting, setDetecting] = useState(false)
  const [detectError, setDetectError] = useState<string | null>(null)

  const [mitigation, setMitigation] = useState<MitigationResult | null>(null)
  const [mitigating, setMitigating] = useState(false)
  const [mitigateError, setMitigateError] = useState<string | null>(null)

  const [metrics, setMetrics] = useState<EvalMetrics | null>(null)
  const [metricsLoading, setMetricsLoading] = useState(true)

  useEffect(() => {
    getEvalMetrics()
      .then(setMetrics)
      .catch(() => setMetrics(null))
      .finally(() => setMetricsLoading(false))
  }, [])

  async function handleSubmit(text: string) {
    if (!text) return
    setDetecting(true)
    setDetectError(null)
    setMitigation(null)
    setMitigateError(null)
    try {
      const data = await detectBias(text)
      setResult(data)
    } catch (err) {
      setDetectError(err instanceof Error ? err.message : 'Detection failed')
      setResult(null)
    } finally {
      setDetecting(false)
    }
  }

  async function handleMitigate() {
    if (!result) return
    setMitigating(true)
    setMitigateError(null)
    try {
      const flaggedCategories = Object.entries(result.categories)
        .filter(([, v]) => v.flagged)
        .map(([category]) => category)
      const flaggedLabel = flaggedCategories.map((category) => categoryLabel(category)).join(', ')
      const data = await mitigateBias(result.text, flaggedLabel || null, flaggedCategories)
      setMitigation(data)
    } catch (err) {
      setMitigateError(err instanceof Error ? err.message : 'Mitigation failed')
    } finally {
      setMitigating(false)
    }
  }

  return (
    <div className="h-full w-full overflow-y-auto bg-bg-0">
      <header className="sticky top-0 z-10 flex items-center gap-2 px-6 py-4 border-b border-border bg-bg-0/85 backdrop-blur-md">
        <Scale size={18} className="text-accent-text" />
        <span className="font-semibold text-text-1">BiasLens</span>
        <span className="text-xs text-text-4">— Vietnamese bias detection</span>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-10 flex flex-col gap-10">
        {STAGES.map((stage) => (
          <section key={stage.n} className="flex gap-4">
            <div className="flex flex-col items-center flex-shrink-0 pt-1">
              <div className="w-7 h-7 rounded-full border border-border-subtle bg-bg-2 flex items-center justify-center text-xs font-semibold text-accent-text">
                {stage.n}
              </div>
              {stage.n < STAGES.length && <div className="flex-1 w-px bg-border mt-2" />}
            </div>

            <div className="flex-1 pb-2">
              <h2 className="text-sm font-semibold text-text-1 mb-3">{stage.title}</h2>

              {stage.n === 1 && <InputStage onSubmit={handleSubmit} loading={detecting} />}
              {stage.n === 2 && (
                <DetectStage result={result} loading={detecting} error={detectError} />
              )}
              {stage.n === 3 && (
                <MitigateStage
                  detection={result}
                  result={mitigation}
                  loading={mitigating}
                  error={mitigateError}
                  onMitigate={handleMitigate}
                />
              )}
              {stage.n === 4 && <EvaluateStage metrics={metrics} loading={metricsLoading} />}
            </div>
          </section>
        ))}
      </main>
    </div>
  )
}
