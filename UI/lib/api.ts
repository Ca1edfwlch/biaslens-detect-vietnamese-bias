import { DetectionResult, EvalMetrics, MitigationResult } from '@/types'

export async function detectBias(text: string): Promise<DetectionResult> {
  const response = await fetch('/api/detect-severity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || 'Detection request failed')
  }

  return response.json()
}

export async function mitigateBias(
  text: string,
  label?: string | null,
  categories?: string[] | null,
): Promise<MitigationResult> {
  const response = await fetch('/api/mitigate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, label, categories }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || 'Mitigation request failed')
  }

  return response.json()
}

export async function getEvalMetrics(): Promise<EvalMetrics | null> {
  const response = await fetch('/api/metrics')
  if (response.status === 404) return null
  if (!response.ok) throw new Error('Failed to load evaluation metrics')
  return response.json()
}
